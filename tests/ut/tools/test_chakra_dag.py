# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""``chakra_dag`` 的离线单元测试。

设计约束：这些测试必须在**没有 torch、没有 NPU、没有 chakra** 的个人电脑上全部通过。
因此：

- 通过 ``sys.path`` 直接加载包，绕开 ``mindspeed_llm/__init__.py`` 的补丁注册副作用；
- 涉及 chakra 的用例一律 ``pytest.importorskip``；
- 覆盖的是纯逻辑：IR 序列化、融合展开、建图、多卡合并、环检测、trace 解析。

真实 trace 的字段校验只能在公司服务器上做，见改动文档的验收清单。
"""

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

PKG_DIR = (
    Path(os.environ["CHAKRA_DAG_PKG"])
    if "CHAKRA_DAG_PKG" in os.environ
    else Path(__file__).resolve().parents[3] / "mindspeed_llm" / "tools" / "chakra_dag"
)


def _load_package():
    """按文件路径加载 ``chakra_dag`` 包，避免触发 mindspeed_llm 的导入副作用。"""
    name = "chakra_dag_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pkg():
    _load_package()
    modules = {}
    for sub in ("ir", "fused_ops", "graph", "timing", "writer", "capture", "export", "hooks", "cli"):
        modules[sub] = importlib.import_module(f"chakra_dag_under_test.{sub}")
    return type("Pkg", (), modules)


def _comm_event(pkg, eid, seq, deps, group="g0", comm_type="all_reduce", nbytes=1024, occurrence=0):
    comm = pkg.ir.CommMeta(
        comm_type=comm_type,
        group_name=group,
        comm_name=group,
        count=nbytes // 2,
        dtype="bf16",
        nbytes=nbytes,
        participants=[0, 1],
        occurrence=occurrence,
    )
    return pkg.ir.Event(
        eid=eid,
        seq=seq,
        kind=pkg.ir.EVENT_COMM_COLL,
        name=f"{comm_type}_{occurrence}",
        deps=deps,
        dep_sources=[pkg.ir.DEP_SEQUENCE] * len(deps),
        comm=comm,
    )


def _comp_event(pkg, eid, seq, deps, name="matmul"):
    return pkg.ir.Event(
        eid=eid,
        seq=seq,
        kind=pkg.ir.EVENT_COMPUTE,
        name=name,
        deps=deps,
        dep_sources=[pkg.ir.DEP_TENSOR] * len(deps),
    )


# ---------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------
def test_event_roundtrip_preserves_fields(pkg):
    event = _comm_event(pkg, "c1", 0, [])
    payload = event.to_dict()
    restored = pkg.ir.Event.from_dict(json.loads(json.dumps(payload)))
    assert restored.eid == event.eid
    assert restored.comm.comm_type == "all_reduce"
    assert restored.comm.nbytes == 1024


def test_event_rejects_unknown_kind(pkg):
    with pytest.raises(ValueError):
        pkg.ir.Event(eid="x", seq=0, kind="not_a_kind", name="n")


def test_comm_match_key_requires_group_and_type(pkg):
    comm = pkg.ir.CommMeta(comm_type="", group_name="", nbytes=8)
    assert comm.match_key() is None
    comm.comm_type = "all_reduce"
    comm.group_name = "g"
    assert comm.match_key() == ("g", "all_reduce", 8, None, 0)


def test_dtype_size_table(pkg):
    assert pkg.ir.dtype_size("bf16") == 2
    assert pkg.ir.dtype_size("float32") == 4
    assert pkg.ir.dtype_size("unknown-dtype") is None


# ---------------------------------------------------------------------------
# 融合算子展开
# ---------------------------------------------------------------------------
def test_fused_rule_expands_comm_and_compute(pkg):
    bound = pkg.fused_ops.BoundArgs(
        op="all_gather_matmul",
        args={
            "input1": pkg.fused_ops.ArgValue(name="input1", numel=100, dtype="fp16"),
            "comm_domain": pkg.fused_ops.ArgValue(name="comm_domain", text="1"),
        },
        context={"comm_domain": "1"},
    )
    spec = pkg.fused_ops.build_spec(bound)
    assert spec is not None
    assert spec.ops == ["all_gather", "matmul"]
    assert spec.comm_type == "all_gather"
    assert spec.nbytes == 200
    assert spec.group_name == "1"


def test_fused_spec_unknown_op_returns_none(pkg):
    bound = pkg.fused_ops.BoundArgs(op="ordinary_matmul", args={})
    assert pkg.fused_ops.build_spec(bound) is None


def test_fused_expansion_produces_siblings_not_chain(pkg):
    """融合体展开后两个节点必须是兄弟：共同前驱、共同后继。"""
    events = [
        _comp_event(pkg, "e0", 0, [], name="rmsnorm"),
        pkg.ir.Event(
            eid="e1",
            seq=1,
            kind=pkg.ir.EVENT_FUSED,
            name="all_gather_matmul",
            deps=["e0"],
            dep_sources=[pkg.ir.DEP_TENSOR],
            fused=pkg.ir.FusedSpec(
                ops=["all_gather", "matmul"],
                comm_type="all_gather",
                group_name="tp0",
                nbytes=2048,
                participants=[0, 1],
            ),
        ),
        _comp_event(pkg, "e2", 2, ["e1"], name="linear_out"),
    ]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    by_name = {node.name: node for node in graph.nodes}
    comm = by_name["all_gather"]
    comp = by_name["matmul"]
    after = by_name["linear_out"]
    assert comm.kind == pkg.ir.COMM_COLL
    assert comp.kind == pkg.ir.COMP
    # 兄弟：同一前驱
    assert comm.preds == comp.preds
    # 后继算子同时依赖两者
    assert set(after.preds) == {comm.id, comp.id}
    # 融合体内部不能互相依赖
    assert comp.id not in comm.preds
    assert comm.id not in comp.preds


def test_fused_without_rule_degrades_but_is_flagged(pkg):
    events = [
        pkg.ir.Event(
            eid="e0",
            seq=0,
            kind=pkg.ir.EVENT_FUSED,
            name="some_new_fused_op",
            deps=[],
            dep_sources=[],
            fused=None,
        )
    ]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].attrs["fused_unexpanded"] is True
    assert graph.meta["stats"]["unresolved_fused"] == ["some_new_fused_op"]


def test_unresolved_comm_candidates_detects_missing_rules(pkg):
    missing = pkg.fused_ops.unresolved_comm_candidates(
        ["npu_all_gather_base_mm", "brand_new_all_gather_mm", "ordinary_mm", "get_hccl_comm_name"]
    )
    assert missing == ["brand_new_all_gather_mm"]


# ---------------------------------------------------------------------------
# 建图
# ---------------------------------------------------------------------------
def test_build_graph_uses_deps_not_timestamps(pkg):
    """依赖必须只来自事件记录：即便 seq 顺序相反，也按 deps 建边。"""
    events = [
        _comp_event(pkg, "a", 0, []),
        _comp_event(pkg, "b", 1, []),  # 与 a 无依赖
        _comp_event(pkg, "c", 2, ["a"]),
    ]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    by_name = {node.seq: node for node in graph.nodes}
    assert by_name[1].preds == []
    assert len(by_name[2].preds) == 1


def test_build_graph_rejects_duplicate_seq(pkg):
    events = [_comp_event(pkg, "a", 0, []), _comp_event(pkg, "b", 0, [])]
    with pytest.raises(pkg.graph.GraphError):
        pkg.graph.build_rank_graph(rank=0, events=events)


def test_succs_are_derived_from_preds(pkg):
    events = [_comp_event(pkg, "a", 0, []), _comp_event(pkg, "b", 1, ["a"])]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    first = graph.nodes[0]
    assert len(first.succs) == 1
    assert graph.nodes[1].id in first.succs


def test_edge_deduplication(pkg):
    events = [_comp_event(pkg, "a", 0, []), _comp_event(pkg, "b", 1, ["a", "a"])]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    assert len(graph.nodes[1].preds) == 1


def test_cycle_detection(pkg):
    graph = pkg.ir.RankGraph(rank=0, nodes=[])
    n1 = pkg.ir.Node(id="n1", kind=pkg.ir.COMP, name="a", rank=0, seq=0)
    n2 = pkg.ir.Node(id="n2", kind=pkg.ir.COMP, name="b", rank=0, seq=1)
    n1.add_pred("n2", pkg.ir.DEP_SEQUENCE)
    n2.add_pred("n1", pkg.ir.DEP_SEQUENCE)
    graph.nodes = [n1, n2]
    graph.index_succs()
    cycles = pkg.graph.find_cycles(graph)
    assert cycles
    assert pkg.graph.validate(graph)["acyclic"] is False


# ---------------------------------------------------------------------------
# 多卡合并
# ---------------------------------------------------------------------------
def _two_rank_graphs(pkg):
    graphs = []
    for rank in (0, 1):
        events = [
            _comp_event(pkg, f"a{rank}", 0, []),
            _comm_event(pkg, f"c{rank}", 1, [f"a{rank}"], occurrence=0),
            _comp_event(pkg, f"b{rank}", 2, [f"c{rank}"]),
        ]
        graphs.append(pkg.graph.build_rank_graph(rank=rank, events=events))
    return graphs


def test_multi_rank_merges_one_collective_node(pkg):
    graphs = _two_rank_graphs(pkg)
    merged, report = pkg.graph.merge_multi_rank(graphs)
    collectives = [node for node in merged.nodes if node.kind == pkg.ir.COMM_COLL]
    assert len(collectives) == 1
    assert report["collectives_merged"] == 1
    assert collectives[0].comm.participants == [0, 1]
    assert collectives[0].attrs["ranks"] == [0, 1]
    assert report["cycles"] == 0


def test_merged_collective_keeps_per_rank_provenance(pkg):
    graphs = _two_rank_graphs(pkg)
    merged, _ = pkg.graph.merge_multi_rank(graphs)
    collective = next(node for node in merged.nodes if node.kind == pkg.ir.COMM_COLL)
    sources = collective.attrs["sources"]
    assert {item["rank"] for item in sources} == {0, 1}


def test_merge_rewires_edges_to_merged_node(pkg):
    graphs = _two_rank_graphs(pkg)
    merged, _ = pkg.graph.merge_multi_rank(graphs)
    collective = next(node for node in merged.nodes if node.kind == pkg.ir.COMM_COLL)
    # 合并节点的前驱应来自两个 rank 的 a 节点，后继应指向两个 b 节点
    assert len(collective.preds) == 2
    assert len(collective.succs) == 2
    # 且图上不应残留指向被合并节点的边
    known = {node.id for node in merged.nodes}
    for node in merged.nodes:
        for pred in node.preds:
            assert pred in known


def test_occurrence_distinguishes_repeated_collectives(pkg):
    """同组同类型但序号不同的两次 collective 不能被误合并。"""
    graphs = []
    for rank in (0, 1):
        events = [
            _comp_event(pkg, f"a{rank}", 0, []),
            _comm_event(pkg, f"c{rank}0", 1, [f"a{rank}"], occurrence=0),
            _comm_event(pkg, f"c{rank}1", 2, [f"c{rank}0"], occurrence=1),
        ]
        graphs.append(pkg.graph.build_rank_graph(rank=rank, events=events))
    merged, report = pkg.graph.merge_multi_rank(graphs)
    assert report["collectives_merged"] == 2


def test_collective_without_group_is_not_merged(pkg):
    graphs = []
    for rank in (0, 1):
        comm = pkg.ir.CommMeta(comm_type="all_reduce", group_name="", nbytes=64)
        events = [
            pkg.ir.Event(
                eid=f"c{rank}",
                seq=0,
                kind=pkg.ir.EVENT_COMM_COLL,
                name="allreduce",
                deps=[],
                dep_sources=[],
                comm=comm,
            )
        ]
        graphs.append(pkg.graph.build_rank_graph(rank=rank, events=events))
    merged, report = pkg.graph.merge_multi_rank(graphs)
    assert report["collectives_merged"] == 0
    assert report["collectives_unmergeable"] == 2


# ---------------------------------------------------------------------------
# 时间与融合拆分
# ---------------------------------------------------------------------------
def test_apply_compute_timings_consumes_in_order(pkg):
    events = [_comp_event(pkg, "a", 0, []), _comp_event(pkg, "b", 1, ["a"])]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    stats = pkg.timing.apply_compute_timings(graph, {"matmul": [(10.0, 5.0), (20.0, 7.0)]})
    assert stats["measured"] == 2
    assert graph.nodes[0].timing.start_us == 10.0
    assert graph.nodes[1].timing.start_us == 20.0


def test_fused_split_estimate_produces_parallel_windows(pkg):
    events = [
        pkg.ir.Event(
            eid="f0",
            seq=0,
            kind=pkg.ir.EVENT_FUSED,
            name="all_gather_matmul",
            deps=[],
            dep_sources=[],
            fused=pkg.ir.FusedSpec(
                ops=["all_gather", "matmul"],
                comm_type="all_gather",
                group_name="tp",
                nbytes=1_000_000,
                participants=[0, 1],
                order="comm_first",
            ),
        )
    ]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    # 模拟实测：整个融合体 1000us
    for node in graph.nodes:
        node.timing = pkg.ir.Timing(start_us=0.0, duration_us=1000.0, source=pkg.ir.TIME_MEASURED)
    stats = pkg.timing.apply_fused_split(graph, mode=pkg.ir.SPLIT_ESTIMATE, default_bandwidth_gbps=10.0)
    comm = next(node for node in graph.nodes if node.kind == pkg.ir.COMM_COLL)
    comp = next(node for node in graph.nodes if node.kind == pkg.ir.COMP)
    assert stats["split_estimated"] == 1
    # 1MB / 10GB/s = 100us
    assert comm.timing.duration_us == pytest.approx(100.0, rel=1e-6)
    assert comm.timing.source == pkg.ir.TIME_ESTIMATED
    # 计算段与通信段都落在包络内
    assert comp.timing.start_us + comp.timing.duration_us == pytest.approx(1000.0, rel=1e-6)


def test_fused_split_falls_back_to_merged_without_bandwidth(pkg):
    events = [
        pkg.ir.Event(
            eid="f0",
            seq=0,
            kind=pkg.ir.EVENT_FUSED,
            name="matmul_all_reduce",
            deps=[],
            dep_sources=[],
            fused=pkg.ir.FusedSpec(
                ops=["matmul", "all_reduce"],
                comm_type="all_reduce",
                group_name="tp",
                nbytes=1000,
                participants=[0, 1],
            ),
        )
    ]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    for node in graph.nodes:
        node.timing = pkg.ir.Timing(start_us=0.0, duration_us=500.0, source=pkg.ir.TIME_MEASURED)
    stats = pkg.timing.apply_fused_split(graph, mode=pkg.ir.SPLIT_ESTIMATE, default_bandwidth_gbps=None)
    assert stats["merged"] == 1
    for node in graph.nodes:
        assert node.attrs["time_split"] == pkg.ir.SPLIT_MERGED


def test_normalize_timeline_shifts_to_zero(pkg):
    events = [_comp_event(pkg, "a", 0, []), _comp_event(pkg, "b", 1, ["a"])]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    graph.nodes[0].timing = pkg.ir.Timing(start_us=1_700_000_000_000.0, duration_us=10.0)
    graph.nodes[1].timing = pkg.ir.Timing(start_us=1_700_000_000_100.0, duration_us=10.0)
    pkg.timing.normalize_timeline(graph)
    assert graph.nodes[0].timing.start_us == 0.0
    assert graph.nodes[1].timing.start_us == 100.0
    assert graph.meta["time_offset_us"] == 1_700_000_000_000.0


# ---------------------------------------------------------------------------
# trace 解析（用合成 fixture，不使用真实数据）
# ---------------------------------------------------------------------------
SAMPLE_TRACE = {
    "traceEvents": [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "Ascend Hardware"}},
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Communication"}},
        {"ph": "M", "name": "process_name", "pid": 3, "args": {"name": "CANN"}},
        {
            "ph": "X",
            "pid": 1,
            "name": "MatMul",
            "ts": 1000,
            "dur": 40,
            "args": {"Task Type": "AI_CORE"},
        },
        {
            "ph": "X",
            "pid": 1,
            "name": "MatMul",
            "ts": 1100,
            "dur": 50,
            "args": {"Task Type": "MIX_AIC"},
        },
        {
            "ph": "X",
            "pid": 1,
            "name": "memcpy_stub",
            "ts": 1200,
            "dur": 1,
            "args": {"Task Type": "AI_CORE"},
        },
        {
            "ph": "X",
            "pid": 1,
            "name": "sys_op",
            "ts": 1300,
            "dur": 3,
            "args": {"Task Type": "EVENT_RECORD"},
        },
        {
            "ph": "X",
            "pid": 2,
            "name": "hcom_allReduce__12345_1_1",
            "ts": 1150,
            "dur": 120,
            "args": {"count": 1024, "data_type": "bf16", "group_name": "g0"},
        },
        {
            "ph": "X",
            "pid": 2,
            "name": "hcom_send__1",
            "ts": 1400,
            "dur": 20,
            "args": {"count": 8},
        },
        {
            "ph": "X",
            "pid": 3,
            "name": "aten::matmul",
            "ts": 999,
            "dur": 1,
            "args": {"cpu_op": True},
        },
    ]
}


def test_extract_compute_timings_filters_layers(pkg):
    result = pkg.timing.extract_compute_timings(SAMPLE_TRACE["traceEvents"])
    assert "MatMul" in result
    assert len(result["MatMul"]) == 2
    assert result["MatMul"][0] == (1000.0, 40.0)
    # Memcpy 前缀、EVENT_RECORD、CANN 层都被排除
    assert "memcpy_stub" not in result
    assert "sys_op" not in result
    assert "aten::matmul" not in result


def test_extract_collective_timings_excludes_p2p(pkg):
    records = pkg.timing.extract_collective_timings(SAMPLE_TRACE["traceEvents"])
    names = [item["name"] for item in records]
    assert names == ["hcom_allReduce__12345_1_1"]
    assert records[0]["comm_type"] == "all_reduce"
    assert records[0]["nbytes"] == 2048


def test_classify_comm_name(pkg):
    assert pkg.timing.classify_comm_name("hcom_allReduce__1") == "all_reduce"
    assert pkg.timing.classify_comm_name("hcom_allGather__1") == "all_gather"
    assert pkg.timing.classify_comm_name("hcom_reduceScatter__1") == "reduce_scatter"
    assert pkg.timing.classify_comm_name("hcom_alltoAll__1") == "all_to_all"
    assert pkg.timing.classify_comm_name("hcom_weird") == ""


def test_resolve_rank_prefers_metadata(pkg):
    assert pkg.timing.resolve_rank({"distributed_args": {"rank": 3}}, "/x/trace_view.json") == 3
    assert pkg.timing.resolve_rank({}, "/root/rank5/ASCEND_PROFILER_OUTPUT/trace_view.json") == 5


def test_translate_group_uses_parallel_group_info(pkg):
    metadata = {"parallel_group_info": {"0": {"group_name": "tp_group", "global_ranks": [0, 1, 2, 3]}}}
    assert pkg.timing.translate_group(metadata, "tp_group") == [0, 1, 2, 3]
    assert pkg.timing.translate_group(metadata, "unknown") is None


def test_trace_only_build_marks_structure(pkg, tmp_path):
    trace_dir = tmp_path / "rank0" / "ASCEND_PROFILER_OUTPUT"
    trace_dir.mkdir(parents=True)
    trace_file = trace_dir / "trace_view.json"
    trace_file.write_text(json.dumps(SAMPLE_TRACE), encoding="utf-8")
    meta_file = tmp_path / "rank0" / "profiler_metadata.json"
    meta_file.write_text(
        json.dumps(
            {
                "distributed_args": {"rank": 0, "world_size": 2},
                "parallel_group_info": {"0": {"group_name": "g0", "global_ranks": [0, 1]}},
            }
        ),
        encoding="utf-8",
    )
    graphs, report = pkg.export.build_graphs_from_traces([str(trace_file)], str(tmp_path))
    assert report["structure"] == "trace_only"
    assert len(graphs) == 1
    assert graphs[0].rank == 0
    collective = next(node for node in graphs[0].nodes if node.kind == pkg.ir.COMM_COLL)
    assert collective.comm.participants == [0, 1]


def test_dedupe_traces_drops_duplicates(pkg, tmp_path):
    content = json.dumps(SAMPLE_TRACE)
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")
    kept, dropped = pkg.export.dedupe_traces([str(first), str(second)])
    assert len(kept) == 1
    assert len(dropped) == 1


# ---------------------------------------------------------------------------
# writer（chakra 可选）
# ---------------------------------------------------------------------------
def test_writer_status_reports_missing_chakra_without_raising(pkg):
    status = pkg.writer.describe_chakra_status()
    assert "available" in status
    if not status["available"]:
        assert status["error"]


def test_chakra_node_type_mapping_is_complete(pkg):
    for kind in pkg.ir.NODE_KINDS:
        assert kind in pkg.writer.KIND_TO_NODETYPE
    for comm_type in pkg.ir.COMM_TYPES:
        assert comm_type in pkg.writer.COMM_TYPE_TO_ENUM


def test_graph_to_json_is_serialisable(pkg):
    events = [_comp_event(pkg, "a", 0, []), _comm_event(pkg, "c", 1, ["a"])]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    payload = pkg.writer.graph_to_json(graph)
    text = json.dumps(payload, ensure_ascii=False)
    assert '"nodes"' in text
    assert payload["schema"] == "mindspeed-llm/dag/v1"


def test_write_and_read_protobuf_roundtrip(pkg, tmp_path):
    if not pkg.writer.describe_chakra_status()["available"]:
        pytest.skip("chakra 未安装，跳过 protobuf 往返测试")
    events = [_comp_event(pkg, "a", 0, []), _comm_event(pkg, "c", 1, ["a"])]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    for node in graph.nodes:
        node.timing = pkg.ir.Timing(start_us=0.0, duration_us=10.0, source=pkg.ir.TIME_MEASURED)
    path = tmp_path / "rank0.et"
    info = pkg.writer.write_chakra_protobuf(graph, str(path))
    assert info["nodes"] == 2
    metadata, nodes = pkg.writer.read_chakra_file(str(path))
    assert metadata is not None
    assert metadata.version == pkg.writer.CHAKRA_VERSION
    assert len(nodes) == 2
    # 依赖必须指向存在的节点
    ids = {node.id for node in nodes}
    for node in nodes:
        for dep in node.data_deps:
            assert dep in ids


def test_attach_trace_timings_fills_nodes(pkg, tmp_path):
    """结构与时长分离：依赖来自采集，时长来自 trace。"""
    profile_root = tmp_path / "profile"
    trace_dir = profile_root / "rank0" / "ASCEND_PROFILER_OUTPUT"
    trace_dir.mkdir(parents=True)
    trace_file = trace_dir / "trace_view.json"
    trace_file.write_text(json.dumps(SAMPLE_TRACE), encoding="utf-8")
    (profile_root / "rank0" / "profiler_metadata.json").write_text(
        json.dumps({"distributed_args": {"rank": 0, "world_size": 1}}), encoding="utf-8"
    )

    events = [_comp_event(pkg, "a", 0, [], name="MatMul"), _comp_event(pkg, "b", 1, ["a"], name="MatMul")]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    report = {}
    pkg.export.attach_trace_timings([graph], str(profile_root), report)

    # 时长来自 trace，且结构不受影响
    assert graph.nodes[0].timing.duration_us == 40.0
    assert graph.nodes[1].timing.duration_us == 50.0
    assert graph.nodes[0].timing.source == pkg.ir.TIME_MEASURED
    # 起点被归一化到 0，绝对基准保留在 meta 里
    assert graph.nodes[0].timing.start_us == 0.0
    assert graph.nodes[1].timing.start_us == 100.0
    assert graph.meta["time_offset_us"] == 1000.0
    # 结构不受时长影响
    assert graph.nodes[0].preds == []
    assert graph.nodes[1].preds == [graph.nodes[0].id]


def test_attach_trace_timings_without_profile_path_warns(pkg, tmp_path):
    events = [_comp_event(pkg, "a", 0, [])]
    graph = pkg.graph.build_rank_graph(rank=0, events=events)
    report = {}
    pkg.export.attach_trace_timings([graph], None, report)
    assert report["warnings"]
    assert graph.nodes[0].timing.source == pkg.ir.TIME_UNKNOWN


def test_transitive_edge_reduction_keeps_diamond(pkg):
    """菱形依赖（兄弟节点有共同前驱）不能被误删。"""
    a = pkg.ir.Node(id="a", kind=pkg.ir.COMP, name="a", rank=0, seq=0)
    b = pkg.ir.Node(id="b", kind=pkg.ir.COMP, name="b", rank=0, seq=1, preds=["a"])
    c = pkg.ir.Node(id="c", kind=pkg.ir.COMP, name="c", rank=0, seq=2, preds=["a"])
    d = pkg.ir.Node(id="d", kind=pkg.ir.COMP, name="d", rank=0, seq=3, preds=["b", "c"])
    graph = pkg.ir.RankGraph(rank=0, nodes=[a, b, c, d])
    graph.index_succs()
    removed = pkg.graph.reduce_transitive_edges(graph)
    assert removed == 0
    assert set(d.preds) == {"b", "c"}


def test_transitive_edge_reduction_removes_redundancy(pkg):
    """既有 B 又有 B 的上游 A 时，A 那条边是冗余的。"""
    a = pkg.ir.Node(id="a", kind=pkg.ir.COMP, name="a", rank=0, seq=0)
    b = pkg.ir.Node(id="b", kind=pkg.ir.COMP, name="b", rank=0, seq=1, preds=["a"])
    c = pkg.ir.Node(id="c", kind=pkg.ir.COMP, name="c", rank=0, seq=2, preds=["a", "b"])
    graph = pkg.ir.RankGraph(rank=0, nodes=[a, b, c])
    graph.index_succs()
    removed = pkg.graph.reduce_transitive_edges(graph)
    assert removed == 1
    assert c.preds == ["b"]


def test_hooks_disabled_by_default(pkg):
    args = type("A", (), {})()
    assert pkg.hooks.begin(args, 0) is None
    assert pkg.hooks.end(args, 0) is None
    assert pkg.hooks.export(args) is None


def test_hooks_default_output_dir(pkg):
    args = type("A", (), {"profile_save_path": "/tmp/p"})()
    assert pkg.hooks.default_output_dir(args) == "/tmp/p/chakra_dag"
    args2 = type("A", (), {})()
    assert pkg.hooks.default_output_dir(args2) == "./profile_dir/chakra_dag"


def test_hooks_export_only_on_rank0(pkg):
    args = type("A", (), {"chakra_dag_export": True, "rank": 3, "world_size": 8})()
    assert pkg.hooks.export(args) is None


def test_hooks_begin_swallows_install_failure(pkg, monkeypatch):
    """安装失败必须返回 None，绝不抛到训练流程里。"""

    def boom(self):
        raise RuntimeError("dispatch unavailable")

    monkeypatch.setattr(pkg.capture.DagCapture, "install", boom)
    args = type("A", (), {"chakra_dag_export": True, "world_size": 1})()
    assert pkg.hooks.begin(args, 0) is None
    assert pkg.hooks.status()["active_ranks"] == []


def test_hooks_end_skips_write_when_no_events(pkg, tmp_path, monkeypatch):
    """空采集不能写出空文件覆盖上一次的有效结果。"""

    class FakeRecorder:
        events = []
        degraded = True

        def uninstall(self):
            pass

        def notes(self):
            return ["no hooks installed"]

        def dump(self, path):
            raise AssertionError("不应调用 dump")

    monkeypatch.setitem(pkg.hooks._ACTIVE, 0, FakeRecorder())
    args = type("A", (), {"chakra_dag_export": True, "profile_save_path": str(tmp_path)})()
    result = pkg.hooks.end(args, 0)
    assert result["skipped"] is True
    assert result["events"] == 0


def test_hooks_begin_not_registered_when_install_fails(pkg, monkeypatch):
    monkeypatch.setattr(pkg.capture.DagCapture, "install", lambda self: False)
    monkeypatch.setattr(pkg.capture.DagCapture, "__init__", lambda self, **kwargs: _init_degraded(self, **kwargs))
    args = type("A", (), {"chakra_dag_export": True, "world_size": 1})()
    assert pkg.hooks.begin(args, 7) is None
    assert 7 not in pkg.hooks._ACTIVE


def _init_degraded(self, **kwargs):
    self.rank = kwargs.get("rank", 0)
    self.events = []
    self.degraded = True


class _FakeTensor:
    """最小化的假张量，只需 id/_version/numel/dtype/shape。"""

    def __init__(self, numel=8, dtype="fp16", version=0):
        self._numel = numel
        self.dtype = dtype
        self._version = version
        self.shape = (numel,)

    def numel(self):
        return self._numel


class _FakeTorch:
    Tensor = _FakeTensor


def test_dispatch_records_data_dependency_through_outputs(pkg):
    """算子生产张量 → 后续算子消费它，必须形成张量依赖而不是仅线程顺序。"""
    recorder = pkg.capture.DagCapture(rank=0, world_size=1)
    recorder._torch = _FakeTorch()

    produced = _FakeTensor(numel=16)
    consumer = _FakeTensor(numel=16)

    eid_a = recorder._on_dispatch_start(_fake_func("mm"), (consumer,), {})
    recorder._on_dispatch_end(eid_a, produced)

    eid_b = recorder._on_dispatch_start(_fake_func("relu"), (produced,), {})
    event_b = next(item for item in recorder.events if item.eid == eid_b)

    assert eid_a in event_b.deps
    assert pkg.ir.DEP_TENSOR in event_b.dep_sources


def _fake_func(name):
    def func(*args, **kwargs):
        return None

    func.__name__ = name
    return func


def test_cli_without_inputs_reports_error(pkg, tmp_path):
    report = pkg.cli.main(["--output", str(tmp_path / "out")])
    assert report["errors"]
