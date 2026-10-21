"""chakra_trace 离线转换脚本的单元测试。

被测文件 mindspeed_llm/tools/chakra_trace.py 只依赖 Python 标准库，
因此这里通过文件路径直接加载它，避免导入 mindspeed_llm 包触发补丁注册。
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "mindspeed_llm" / "tools" / "chakra_trace.py"
SAMPLE_TRACE_PATH = Path(__file__).resolve().parent / "sample_trace_view.json"


def load_module():
    """按文件路径加载 chakra_trace 模块。"""
    spec = importlib.util.spec_from_file_location("chakra_trace", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chakra_trace():
    return load_module()


def make_compute_event(name="MatMul", ts=1000, dur=50, stream=1, pid=1):
    """构造一个计算事件；stream 为 None 时不写 args，用于模拟缺少流信息的事件。"""
    event = {"name": name, "cat": "Kernel", "ph": "X", "ts": ts, "dur": dur, "pid": pid, "tid": 1}
    if stream is not None:
        event["args"] = {"stream": stream}
    return event


def test_load_trace_round_trip(chakra_trace, tmp_path):
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps({"traceEvents": [make_compute_event()]}))
    trace = chakra_trace.load_trace(trace_file)
    assert len(trace["traceEvents"]) == 1


def test_load_trace_rejects_bad_input(chakra_trace, tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json")
    with pytest.raises(ValueError):
        chakra_trace.load_trace(bad_file)
    list_file = tmp_path / "list.json"
    list_file.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        chakra_trace.load_trace(list_file)


def test_classify_event(chakra_trace):
    hccl = chakra_trace.classify_event({"name": "HCCL_ALLREDUCE", "cat": "kernel"})
    assert hccl == {"kind": "communication", "communication_type": "hccl"}
    compute = chakra_trace.classify_event({"name": "MatMul", "cat": "Kernel"})
    assert compute["kind"] == "compute"
    other = chakra_trace.classify_event({"name": "process_name", "cat": "metadata"})
    assert other["kind"] == "other"


def test_extract_events_fields(chakra_trace):
    trace = {
        "traceEvents": [
            make_compute_event(name="MatMul", ts=1000, dur=50, stream=7),
            {"name": "bad_ts", "cat": "Kernel", "ph": "X", "ts": "abc", "dur": "x", "pid": 1, "tid": 1},
            {"name": "no_args", "cat": "Kernel", "ph": "X", "ts": 2000, "dur": 10, "pid": 1, "tid": 1},
        ]
    }
    events = chakra_trace.extract_events(trace)
    assert events[0]["start_time_micros"] == 1000
    assert events[0]["duration_micros"] == 50
    assert events[0]["stream_id"] == 7
    assert events[1]["start_time_micros"] is None
    assert events[1]["duration_micros"] == 0
    assert events[2]["stream_id"] is None


def test_build_dependencies_same_stream(chakra_trace):
    events = chakra_trace.extract_events(
        {"traceEvents": [make_compute_event(name="A", ts=1000), make_compute_event(name="B", ts=1100)]}
    )
    dependency_events, warnings = chakra_trace.build_dependencies(events)
    assert warnings == []
    # node_id 是 source_index + 1，B(source_index=1) 依赖 A(source_index=0)
    assert dependency_events[1]["data_dependencies"] == [1]
    assert dependency_events[1]["dependency_reasons"] == {"1": "same_stream"}


def test_build_dependencies_missing_stream(chakra_trace):
    events = chakra_trace.extract_events(
        {"traceEvents": [make_compute_event(name="A", ts=1000, stream=None), make_compute_event(name="B", ts=1100)]}
    )
    dependency_events, warnings = chakra_trace.build_dependencies(events)
    assert dependency_events[0]["data_dependencies"] == []
    assert len(warnings) == 1
    assert "no process or stream information" in warnings[0]


def test_encode_varint(chakra_trace):
    assert chakra_trace.encode_varint(0) == b"\x00"
    assert chakra_trace.encode_varint(127) == b"\x7f"
    assert chakra_trace.encode_varint(128) == b"\x80\x01"
    assert chakra_trace.encode_varint(300) == b"\xac\x02"


def read_chakra_messages(output_path):
    """按 varint 长度前缀逐条读取 Chakra 文件，返回 message 列表。"""
    from chakra.schema.protobuf import et_def_pb2

    data = Path(output_path).read_bytes()
    messages = []
    offset = 0
    while offset < len(data):
        size = 0
        shift = 0
        while True:
            byte = data[offset]
            offset += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        payload = data[offset : offset + size]
        offset += size
        if not messages:
            message = et_def_pb2.GlobalMetadata()
        else:
            message = et_def_pb2.Node()
        message.ParseFromString(payload)
        messages.append(message)
    return messages


def test_sample_trace_statistics(chakra_trace):
    """对样例 trace 文件执行统计和依赖分析，不依赖 Chakra 包。"""
    trace = chakra_trace.load_trace(SAMPLE_TRACE_PATH)
    dependency_events, warnings = chakra_trace.build_dependencies(chakra_trace.extract_events(trace))
    summary = chakra_trace.summarize_trace(trace, dependency_events, warnings)
    assert summary["event_count"] == 13
    assert summary["dependency_edge_count"] == 5
    assert summary["dependency_reasons"] == {"correlation": 1, "flow": 1, "same_stream": 3}
    assert len(warnings) == 2


def test_sample_trace_export(chakra_trace, tmp_path):
    """把样例 trace 文件导出成 Chakra protobuf 并读回校验节点类型和依赖。"""
    et_def_pb2 = pytest.importorskip("chakra.schema.protobuf.et_def_pb2")
    output_file = tmp_path / "sample.chakra"
    summary = chakra_trace.export_chakra_trace(SAMPLE_TRACE_PATH, output_file)
    assert summary["chakra_node_count"] == 10

    messages = read_chakra_messages(output_file)
    assert messages[0].version == chakra_trace.CHAKRA_SCHEMA_VERSION
    nodes = messages[1:]
    node_types = {node.name: node.type for node in nodes}
    assert node_types["aten::matmul"] == et_def_pb2.COMP_NODE
    assert node_types["hcom_allReduce__123_1"] == et_def_pb2.COMM_COLL_NODE
    assert node_types["HcomSend"] == et_def_pb2.COMM_SEND_NODE
    assert node_types["HcomReceive"] == et_def_pb2.COMM_RECV_NODE
    # HcomSend 与 HcomReceive 同流，后者依赖前者
    send = next(node for node in nodes if node.name == "HcomSend")
    receive = next(node for node in nodes if node.name == "HcomReceive")
    assert list(receive.data_deps) == [send.id]


def test_export_chakra_json(chakra_trace, tmp_path):
    """--format json 应写出单个 JSON 文档，字段命名与 chakra_jsonizer 一致。"""
    pytest.importorskip("chakra")
    output_file = tmp_path / "out.json"
    summary = chakra_trace.export_chakra_trace(SAMPLE_TRACE_PATH, output_file, output_format="json")
    assert summary["chakra_node_count"] == 10

    document = json.loads(output_file.read_text())
    assert document["metadata"]["version"] == chakra_trace.CHAKRA_SCHEMA_VERSION
    assert len(document["nodes"]) == 10
    node_types = {node["name"]: node["type"] for node in document["nodes"]}
    assert node_types["MatMul"] == "COMP_NODE"
    assert node_types["HcomReceive"] == "COMM_RECV_NODE"


def test_export_chakra_trace_round_trip(chakra_trace, tmp_path):
    et_def_pb2 = pytest.importorskip("chakra.schema.protobuf.et_def_pb2")
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(
        json.dumps(
            {
                "traceEvents": [
                    make_compute_event(name="MatMul", ts=1000, dur=50, stream=1),
                    {
                        "name": "HCCL_ALLREDUCE",
                        "cat": "kernel",
                        "ph": "X",
                        "ts": 1050,
                        "dur": 80,
                        "pid": 1,
                        "tid": 1,
                        "args": {"stream": 2},
                    },
                    make_compute_event(name="Scatter", ts=1130, dur=20, stream=1),
                    {"name": "HcomSend", "cat": "communication", "ph": "X", "ts": 1160, "dur": 10, "pid": 1, "tid": 2},
                    {
                        "name": "HcomReceive",
                        "cat": "communication",
                        "ph": "X",
                        "ts": 1160,
                        "dur": 12,
                        "pid": 1,
                        "tid": 2,
                    },
                    {"name": "process_name", "cat": "metadata", "ph": "M", "pid": 1, "tid": 0, "args": {"name": "npu"}},
                    {"name": "no_time", "cat": "kernel", "ph": "X", "pid": 1, "tid": 1, "args": {"stream": 1}},
                ]
            }
        )
    )
    output_file = tmp_path / "out.chakra"
    summary = chakra_trace.export_chakra_trace(trace_file, output_file)
    assert summary["chakra_node_count"] == 5

    messages = read_chakra_messages(output_file)
    metadata = messages[0]
    assert metadata.version == chakra_trace.CHAKRA_SCHEMA_VERSION
    metadata_attrs = {attr.name: attr.string_val for attr in metadata.attr}
    assert metadata_attrs["schema"] == "Ascend"

    nodes = messages[1:]
    node_types = {node.name: node.type for node in nodes}
    assert node_types["MatMul"] == et_def_pb2.COMP_NODE
    assert node_types["HCCL_ALLREDUCE"] == et_def_pb2.COMM_COLL_NODE
    assert node_types["HcomSend"] == et_def_pb2.COMM_SEND_NODE
    assert node_types["HcomReceive"] == et_def_pb2.COMM_RECV_NODE
    # Scatter 与 MatMul 同流且更晚开始，应依赖 MatMul（节点号 1）
    scatter = next(node for node in nodes if node.name == "Scatter")
    assert list(scatter.data_deps) == [1]
