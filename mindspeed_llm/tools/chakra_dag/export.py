# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""导出编排：定位 trace、合并、补时长、写产物。

本模块是训练期钩子和离线命令行共用的入口，两条路径走同一套逻辑：

- 训练期：``prof.stop()`` 之后由 ``maybe_export()`` 调起，在**子进程**里执行，
  避免解析上百 MB 的 trace 拖慢训练进程、避免长时间 GC 卡住集合通信；
- 离线：``python -m mindspeed_llm.tools.chakra_dag.cli`` 直接调用。

## 产物

::

    <输出目录>/
    ├── events_rank<N>.json        采集期产物（运行时结构）
    ├── rank<N>.et                 Chakra execution trace（每 rank 一份）
    ├── multi_rank.et              合并图（可选，跨卡 collective 已合并）
    ├── multi_rank_dag.json        合并图 JSON（可读、可 diff、带边来源）
    ├── manifest.json              rank/device/组翻译/时间基准
    └── export_report.json         统计、未匹配项、缺失项、估算占比

## 失败策略

**导出永远不能中断训练。** 所有异常在这里被捕获并写进报告；调用方只看返回字典。
"""

import os
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import graph as graph_mod
from . import ir, timing, writer


def find_trace_files(root: str) -> List[str]:
    """在 profiling 输出根目录下寻找所有 ``trace_view.json``。

    昇腾目录结构通常为::

        <root>/<host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json

    多 rank 时会有多个这样的子树。这里做有界遍历（限制深度），避免在巨大的
    profiling 目录里扫穿。
    """
    found: List[str] = []
    root = os.path.abspath(root)
    for current, dirs, files in os.walk(root):
        depth = current[len(root) :].count(os.sep)
        if depth > 4:
            dirs[:] = []
            continue
        for name in files:
            if name == "trace_view.json":
                found.append(os.path.join(current, name))
    found.sort()
    return found


def find_metadata_files(trace_path: str) -> Optional[str]:
    """从 trace 路径往上找 ``profiler_metadata.json``。"""
    current = os.path.dirname(os.path.abspath(trace_path))
    for _ in range(5):
        candidate = os.path.join(current, "profiler_metadata.json")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def dedupe_traces(paths: Sequence[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """按内容哈希去重。

    实测中同一份 trace 会在目录里出现多份副本（例如 rank0 的多份拷贝）。不去重会
    让同一个 rank 的节点翻倍，跨卡合并也会失败。
    """
    import hashlib

    kept: List[str] = []
    dropped: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    for path in paths:
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            key = digest.hexdigest()
        except OSError:
            key = os.path.abspath(path)
        if key in seen:
            dropped.append({"path": path, "duplicate_of": seen[key]})
            continue
        seen[key] = path
        kept.append(path)
    return kept, dropped


def build_graphs_from_traces(
    trace_paths: Sequence[str],
    output_dir: str,
    fused_mode: str = ir.SPLIT_ESTIMATE,
    default_bandwidth_gbps: Optional[float] = None,
    require_participants: bool = False,
) -> Tuple[List[ir.RankGraph], Dict[str, Any]]:
    """由 trace 文件构建各 rank 的图（纯测量路径）。

    这条路径**只用于两种情况**：
        1. 没有运行期采集，只能用 trace 补图（此时报告里会标注
           ``structure="trace_only"``，跨卡 collective 配对不可靠）；
        2. 校验：把运行期建图的结果与 trace 统计对照。

    正常训练流程应走 ``build_graphs_from_events``。
    """
    report: Dict[str, Any] = {"structure": "trace_only", "ranks": [], "warnings": []}
    graphs: List[ir.RankGraph] = []
    kept, dropped = dedupe_traces(trace_paths)
    report["duplicates"] = dropped

    for trace_path in kept:
        metadata_path = find_metadata_files(trace_path)
        metadata: Dict[str, Any] = {}
        if metadata_path:
            metadata = timing.load_profiler_metadata(metadata_path)
        else:
            report["warnings"].append(f"缺少 profiler_metadata.json: {trace_path}")
        rank = timing.resolve_rank(metadata, trace_path)
        if rank is None:
            report["warnings"].append(f"无法确定 rank，跳过: {trace_path}")
            continue

        events = timing.load_trace_events(trace_path)
        compute_map = timing.extract_compute_timings(events)
        collectives = timing.extract_collective_timings(events)

        build_events: List[ir.Event] = []
        seq = 0
        for name, series in compute_map.items():
            for start, dur in series:
                build_events.append(
                    ir.Event(
                        eid=f"tc{seq}",
                        seq=seq,
                        kind=ir.EVENT_COMPUTE,
                        name=name,
                        deps=[build_events[-1].eid] if build_events else [],
                        dep_sources=[ir.DEP_SEQUENCE] if build_events else [],
                        attrs={"from_trace": True, "trace_ts": start, "trace_dur": dur},
                    )
                )
                seq += 1
        for record in collectives:
            participants = timing.translate_group(metadata, record.get("group_name") or "")
            comm = ir.CommMeta(
                comm_type=record["comm_type"],
                group_name=record.get("group_name") or "",
                count=record.get("count"),
                dtype=record.get("dtype"),
                nbytes=record.get("nbytes"),
                participants=participants,
            )
            build_events.append(
                ir.Event(
                    eid=f"tk{seq}",
                    seq=seq,
                    kind=ir.EVENT_COMM_COLL,
                    name=record["name"],
                    deps=[build_events[-1].eid] if build_events else [],
                    dep_sources=[ir.DEP_INFERRED] if build_events else [],
                    comm=comm,
                    attrs={"from_trace": True, "inferred_structure": True},
                )
            )
            seq += 1

        rank_graph = graph_mod.build_rank_graph(
            rank=rank,
            events=build_events,
            base_ts_us=None,
            meta={"trace_path": trace_path, "metadata_path": metadata_path, "rank_meta": metadata},
        )
        stats = timing.apply_compute_timings(rank_graph, compute_map)
        stats_c = timing.apply_collective_timings(rank_graph, collectives)
        rank_graph.meta["timing_stats"] = {"compute": stats, "collective": stats_c}
        timing.apply_fused_split(rank_graph, mode=fused_mode, default_bandwidth_gbps=default_bandwidth_gbps)

        problems = graph_mod.validate(rank_graph, require_participants=require_participants)
        report["ranks"].append({"rank": rank, "trace": trace_path, "validation": problems})
        graphs.append(rank_graph)

    graphs.sort(key=lambda item: item.rank)
    return graphs, report


def build_graphs_from_events(
    event_paths: Sequence[str],
    fused_mode: str = ir.SPLIT_ESTIMATE,
    default_bandwidth_gbps: Optional[float] = None,
) -> Tuple[List[ir.RankGraph], Dict[str, Any]]:
    """由运行期采集的 ``events_rank<N>.json`` 构建各 rank 的图。

    这是**主路径**：依赖来自运行时观测，不受时间戳启发式影响。
    """
    report: Dict[str, Any] = {"structure": "runtime_capture", "ranks": [], "warnings": []}
    graphs: List[ir.RankGraph] = []
    for path in sorted(event_paths):
        events, meta = ir.load_events(path)
        rank = int(meta.get("rank", 0))
        rank_graph = graph_mod.build_rank_graph(
            rank=rank,
            events=events,
            base_ts_us=meta.get("base_ts_us"),
            meta={"capture_meta": meta, "events_path": path},
        )
        # 这里不做 fused split：拆分需要实测包络，而包络来自 trace，
        # 要等 attach_trace_timings 拿到时长之后再切。此处只建结构。
        timing.normalize_timeline(rank_graph)
        problems = graph_mod.validate(rank_graph)
        report["ranks"].append(
            {
                "rank": rank,
                "events": len(events),
                "capture_degraded": bool(meta.get("degraded")),
                "capture_notes": meta.get("notes", []),
                "validation": problems,
            }
        )
        graphs.append(rank_graph)
    graphs.sort(key=lambda item: item.rank)
    return graphs, report


def _find_trace_for_rank(profile_root: str, rank: int, metadata_rank_path: Optional[str] = None) -> Optional[str]:
    """在 profiling 根目录里定位某个 rank 的 ``trace_view.json``。

    多 rank 时目录通常形如 ``<root>/rank<N>/...`` 或 ``<root>/<host>_<pid>_<ts>_ascend_pt/``，
    无法只靠目录名判定。因此逐个候选读 ``profiler_metadata.json`` 里的
    ``distributed_args.rank``，只有匹配上才采用；读不到 metadata 时退回目录名约定。
    """
    if metadata_rank_path and os.path.exists(metadata_rank_path):
        return metadata_rank_path
    if not profile_root or not os.path.isdir(profile_root):
        return None
    for trace_path in find_trace_files(profile_root):
        metadata_path = find_metadata_files(trace_path)
        if metadata_path:
            try:
                metadata = timing.load_profiler_metadata(metadata_path)
            except (ValueError, OSError):
                metadata = {}
            if timing.resolve_rank(metadata, trace_path) == rank:
                return trace_path
            continue
        # 没有 metadata 时按目录名 rankN 兜底。
        if f"{os.sep}rank{rank}{os.sep}" in trace_path:
            return trace_path
    return None


def attach_trace_timings(
    graphs: Sequence[ir.RankGraph],
    profile_root: Optional[str],
    report: Dict[str, Any],
    bandwidth_gbps: Optional[float] = None,
    fused_mode: str = ir.SPLIT_ESTIMATE,
) -> None:
    """把 trace 的实测时长贴到运行时建好的图上。

    这是"结构来自运行时、时长来自 profiling"这条分工的落地点：

    - 结构（依赖边）完全不动，只在匹配到的节点上填 ``timing``；
    - 匹配不上的节点保持 ``unknown``，并在报告里计数，绝不按时间猜；
    - 融合算子先按实测包络贴好，再按 ``fused_mode`` 做内部拆分。

    副作用：会为每个 rank 把 ``trace_path`` / ``timing_stats`` 写进 ``graph.meta``。
    """
    if not profile_root:
        report.setdefault("warnings", []).append("未提供 --profile-path，节点时长将全部为 unknown（结构仍然有效）。")
        return

    for rank_graph in graphs:
        trace_path = _find_trace_for_rank(profile_root, rank_graph.rank)
        if trace_path is None:
            report.setdefault("warnings", []).append(
                f"rank {rank_graph.rank} 没有找到对应的 trace_view.json，时长留空。"
            )
            continue
        rank_graph.meta["trace_path"] = trace_path
        try:
            events = timing.load_trace_events(trace_path)
        except (ValueError, OSError) as error:
            report.setdefault("warnings", []).append(f"rank {rank_graph.rank} 读取 trace 失败: {error}")
            continue

        base_ts = rank_graph.meta.get("capture_meta", {}).get("base_ts_us") if rank_graph.meta else None
        compute_map = timing.extract_compute_timings(events, base_ts_us=base_ts)
        collectives = timing.extract_collective_timings(events, base_ts_us=base_ts)

        compute_stats = timing.apply_compute_timings(rank_graph, compute_map)
        collective_stats = timing.apply_collective_timings(rank_graph, collectives)
        # 融合体的包络：展开出的多个节点如果都没单独测到，用同类 kernel 的实测值兜底，
        # 再交给 apply_fused_split 决定怎么切。
        _fill_fused_envelope(rank_graph, compute_map)
        fused_stats = timing.apply_fused_split(rank_graph, mode=fused_mode, default_bandwidth_gbps=bandwidth_gbps)
        timing.normalize_timeline(rank_graph)

        rank_graph.meta["timing_stats"] = {
            "compute": compute_stats,
            "collective": collective_stats,
            "fused": fused_stats,
        }
        report.setdefault("timing", []).append(
            {
                "rank": rank_graph.rank,
                "trace": trace_path,
                "compute": compute_stats,
                "collective": collective_stats,
                "fused": fused_stats,
            }
        )


def _fill_fused_envelope(rank_graph: ir.RankGraph, compute_map: Dict[str, List[Tuple[float, float]]]) -> None:
    """给融合体里还没时间的节点补上实测包络。

    融合算子在 trace 里通常表现为一个整体 kernel（名字可能与运行时记录的算子名
    不完全一致，例如 ``AllGatherMM``）。这里按"名字互相包含"做宽松匹配，
    匹配到就把同一段时长给该融合体的所有节点当包络，交给后续拆分逻辑处理。
    """
    groups: Dict[str, List[ir.Node]] = {}
    for node in rank_graph.nodes:
        group = node.attrs.get("fused_group")
        if group:
            groups.setdefault(group, []).append(node)
    for group_id, members in groups.items():
        if any(node.timing.duration_us is not None for node in members):
            continue
        op_names = members[0].attrs.get("fused_ops") or [members[0].name]
        candidates: List[Tuple[float, float]] = []
        for op_name in op_names:
            for kernel_name, series in compute_map.items():
                lowered_kernel = kernel_name.lower()
                if any(part.lower() in lowered_kernel for part in str(op_name).split("_")):
                    candidates.extend(series)
                    break
        if not candidates:
            continue
        candidates.sort(key=lambda pair: pair[0])
        start, dur = candidates[0]
        for node in members:
            node.timing = ir.Timing(start_us=start, duration_us=dur, source=ir.TIME_MEASURED)


def collect_rank_events(output_dir: str) -> List[str]:
    """收集输出目录里的 ``events_rank*.json``。"""
    result: List[str] = []
    if not os.path.isdir(output_dir):
        return result
    for name in sorted(os.listdir(output_dir)):
        if name.startswith("events_rank") and name.endswith(".json"):
            result.append(os.path.join(output_dir, name))
    return result


def write_outputs(
    graphs: Sequence[ir.RankGraph],
    output_dir: str,
    report: Dict[str, Any],
    export_format: str = "both",
    merge_multi_rank: bool = True,
    metadata_extra: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """写出全部产物，并补齐报告。

    ``export_format``：
        - ``protobuf``：只写 ``.et``；
        - ``json``：只写合并图 JSON（不需要 chakra）；
        - ``both``：两者都写。
    """
    os.makedirs(output_dir, exist_ok=True)
    report.setdefault("outputs", [])

    chakra_status = writer.describe_chakra_status()
    report["chakra"] = chakra_status
    want_protobuf = export_format in ("protobuf", "both")
    if want_protobuf and not chakra_status["available"]:
        report.setdefault("warnings", []).append("chakra 不可用，已跳过 protobuf 输出；JSON 输出不受影响。")
        want_protobuf = False

    metadata_attrs = {
        "world_size": metadata_extra.get("world_size") if metadata_extra else None,
        "start_ts": metadata_extra.get("base_ts_us") if metadata_extra else None,
        "profile_step_start": metadata_extra.get("profile_step_start") if metadata_extra else None,
        "profile_step_end": metadata_extra.get("profile_step_end") if metadata_extra else None,
    }

    if want_protobuf:
        for rank_graph in graphs:
            path = os.path.join(output_dir, f"rank{rank_graph.rank}.et")
            try:
                info = writer.write_chakra_protobuf(rank_graph, path, {**metadata_attrs, "rank": rank_graph.rank})
                report["outputs"].append(info)
            except Exception as error:  # pylint: disable=broad-except
                report.setdefault("errors", []).append(f"rank{rank_graph.rank}.et: {error}")

    merged = None
    merge_report: Dict[str, Any] = {}
    if merge_multi_rank and len(graphs) > 1:
        try:
            merged, merge_report = graph_mod.merge_multi_rank(graphs, strict=strict)
        except graph_mod.GraphError as error:
            report.setdefault("errors", []).append(f"合并失败: {error}")
            merge_report = {"error": str(error)}
    report["merge"] = merge_report

    if merged is not None:
        merged.meta["world_size"] = len(graphs)
        merged.meta["chakra"] = chakra_status
        # 汇总只写进产物和报告的 summary 字段，不动调用方传进来的键。
        summary = _summary(merged)
        report["summary"] = summary
        if want_protobuf:
            path = os.path.join(output_dir, "multi_rank.et")
            try:
                report["outputs"].append(writer.write_chakra_protobuf(merged, path, {**metadata_attrs, "merged": True}))
            except Exception as error:  # pylint: disable=broad-except
                report.setdefault("errors", []).append(f"multi_rank.et: {error}")
        if export_format in ("json", "both"):
            payload = writer.graph_to_json(merged, extra={"report_summary": summary})
            report["outputs"].append(writer.write_json(payload, os.path.join(output_dir, "multi_rank_dag.json")))
    elif export_format in ("json", "both"):
        # 单 rank 也输出一份，保持产物一致，便于工具链统一处理。
        for rank_graph in graphs:
            payload = writer.graph_to_json(rank_graph, extra={"report_summary": _summary(rank_graph)})
            report["outputs"].append(
                writer.write_json(payload, os.path.join(output_dir, f"rank{rank_graph.rank}_dag.json"))
            )

    # 汇总摘要只放进报告（写进 JSON 的 extra），不动用户传入的 report 键，
    # 避免调用方传进来的 warning/timing 等字段被覆盖。
    report["outputs"].append(writer.write_json(report, os.path.join(output_dir, "export_report.json")))
    return report


def _summary(graph: ir.RankGraph) -> Dict[str, Any]:
    kinds = graph_mod.count_kinds(graph.nodes)
    sources: Dict[str, int] = {}
    for node in graph.nodes:
        for source in node.pred_sources.values():
            sources[source] = sources.get(source, 0) + 1
    time_sources: Dict[str, int] = {}
    for node in graph.nodes:
        time_sources[node.timing.source] = time_sources.get(node.timing.source, 0) + 1
    return {
        "nodes": len(graph.nodes),
        "edges": sum(len(node.preds) for node in graph.nodes),
        "kinds": kinds,
        "edge_sources": sources,
        "time_sources": time_sources,
        "acyclic": graph_mod.validate(graph)["acyclic"],
    }


def write_manifest(graphs: Sequence[ir.RankGraph], output_dir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """写 ``manifest.json``：记录 rank 与 trace、时间基准、组翻译结果。

    为什么单独一个文件：节点里的时间是相对量，要还原绝对时间必须有基准；
    另外跨卡分析需要知道每个 rank 对应哪个 trace、哪些组翻译成功。
    """
    payload: Dict[str, Any] = {"schema": "mindspeed-llm/dag-manifest/v1", "ranks": []}
    for rank_graph in graphs:
        rank_meta = rank_graph.meta.get("rank_meta") or {}
        payload["ranks"].append(
            {
                "rank": rank_graph.rank,
                "trace_path": rank_graph.meta.get("trace_path"),
                "events_path": rank_graph.meta.get("events_path"),
                "base_ts_us": rank_graph.base_ts_us,
                "time_offset_us": rank_graph.meta.get("time_offset_us"),
                "device_id": _device_id(rank_meta),
                "parallel_groups": _group_table(rank_meta),
                "capture_meta": rank_graph.meta.get("capture_meta"),
            }
        )
    if extra:
        payload["extra"] = extra
    path = os.path.join(output_dir, "manifest.json")
    payload["chakra"] = writer.describe_chakra_status()
    writer.write_json(payload, path)
    return path


def _device_id(rank_meta: Dict[str, Any]) -> Optional[int]:
    for key in ("device_id", "DeviceId", "logic_device_id"):
        value = rank_meta.get(key)
        if isinstance(value, int):
            return value
    distributed = rank_meta.get("distributed_args") or {}
    value = distributed.get("device_id")
    return value if isinstance(value, int) else None


def _group_table(rank_meta: Dict[str, Any]) -> Dict[str, List[int]]:
    table: Dict[str, List[int]] = {}
    for key in ("parallel_group_info", "group_info"):
        source = rank_meta.get(key)
        if not isinstance(source, dict):
            continue
        for entry in source.values():
            if not isinstance(entry, dict):
                continue
            name = entry.get("group_name")
            ranks = entry.get("global_ranks") or entry.get("group_global_rank")
            if name and isinstance(ranks, list):
                table[str(name)] = [int(item) for item in ranks]
    return table


# ---------------------------------------------------------------------------
# 子进程执行：避免训练进程被解析开销拖住
# ---------------------------------------------------------------------------
def run_export_subprocess(argv: Sequence[str], timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
    """在独立 Python 进程里跑导出。

    为什么必须隔离：解析上百 MB 的 JSON 会产生大量短命对象，训练进程里这会引发
    长时间 GC，而 GC 会插入到集合通信的间隙中，导致 collective 超时。子进程即使
    崩了也不影响训练。

    ``timeout_seconds`` 到点后强制结束，同样只影响导出。
    """
    command = [sys.executable, "-m", "mindspeed_llm.tools.chakra_dag.cli", *argv]
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "seconds": time.time() - started, "command": command}
    except OSError as error:
        return {"ok": False, "error": str(error), "command": command}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "seconds": time.time() - started,
        "command": command,
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
    }


def maybe_export(args, rank: int, world_size: int) -> Optional[Dict[str, Any]]:
    """训练期的唯一入口。

    调用点（见改动文档）：
        - MCore：``mindspeed_llm/training/training.py`` 的 ``prof.stop()`` 之后；
        - FSDP2：``mindspeed_llm/fsdp2/utils/profiler.py`` 的 ``ProfilerManager.stop()`` 之后。

    行为：
        1. 未开启开关、或本 rank 未被采集 → 直接返回 None，零开销；
        2. ``--chakra-dag-inplace`` → 在当前进程导出（调试用）；
        3. 否则起子进程导出，并打印一行结果；
        4. **任何异常都不抛出**，只记日志。
    """
    if not getattr(args, "chakra_dag_export", False):
        return None
    ranks = getattr(args, "chakra_dag_ranks", None)
    if ranks not in (None, [-1], -1) and rank not in (ranks or []):
        return None

    output_dir = getattr(args, "chakra_dag_output_path", None) or os.path.join(
        getattr(args, "profile_save_path", "./profile_dir"), "chakra_dag"
    )
    export_format = getattr(args, "chakra_dag_format", "both")
    fused_mode = getattr(args, "chakra_dag_fused_mode", ir.SPLIT_ESTIMATE)
    bandwidth = getattr(args, "chakra_dag_bandwidth_gbps", None)
    inplace = bool(getattr(args, "chakra_dag_inplace", False))
    timeout = float(getattr(args, "chakra_dag_timeout_seconds", 1800))

    argv = [
        "--output",
        output_dir,
        "--format",
        export_format,
        "--fused-mode",
        fused_mode,
        "--profile-path",
        getattr(args, "profile_save_path", "./profile_dir"),
    ]
    if bandwidth is not None:
        argv.extend(["--bandwidth-gbps", str(bandwidth)])
    source = getattr(args, "chakra_dag_source", "auto")
    argv.extend(["--source", source])

    try:
        if inplace:
            from . import cli  # pylint: disable=import-outside-toplevel

            return {"ok": True, "mode": "inplace", "report": cli.main(argv)}
        result = run_export_subprocess(argv, timeout_seconds=timeout)
        return result
    except Exception:  # pylint: disable=broad-except
        # 导出失败不能影响训练，这是硬约束。
        return {"ok": False, "error": traceback.format_exc()}
