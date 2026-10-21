"""读取昇腾 Profiler 生成的 trace，作为 Chakra 转换的输入。

本脚本只使用 Python 标准库，因此直接运行时不会导入训练包，也不会启动训练。
请在项目根目录运行：
    python mindspeed_llm/tools/chakra_trace.py --input path/to/trace_view.json
"""

import argparse
import json
from pathlib import Path


def load_trace(input_path):
    """读取并检查 trace 文件。

    输入：
        input_path：trace_view.json 文件路径，可以是字符串或 Path 对象。

    输出：
        返回解析后的 trace 字典，保留输入 JSON 中的全部字段。

    处理内容：
        读取 JSON 文件，检查顶层对象、traceEvents 列表以及每个事件的基本类型。
        如果文件不存在、JSON 格式错误或结构不符合要求，则抛出异常。
    """
    with Path(input_path).open("r", encoding="utf-8") as trace_file:
        trace = json.load(trace_file)

    if not isinstance(trace, dict):
        raise ValueError("Trace must be a JSON object containing traceEvents.")
    if "traceEvents" not in trace:
        raise ValueError("Trace is missing traceEvents.")

    events = trace["traceEvents"]
    if not isinstance(events, list):
        raise ValueError("traceEvents must be a list.")

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"traceEvents[{index}] must be an object.")
        for field in ("cat", "ph"):
            if field in event and not isinstance(event[field], str):
                raise ValueError(f"traceEvents[{index}].{field} must be a string.")

    return trace


def classify_event(event):
    """判断一个事件的基本类型。

    输入：
        event：traceEvents 列表中的一个事件字典。

    输出：
        返回一个字典，包含：
        - kind：事件类型，值为 compute、communication 或 other。
        - communication_type：通信类型；HCCL 事件为 hccl，其他通信事件为 unknown，
          非通信事件为空字符串。

    处理内容：
        优先根据事件名称和类别中的 HCCL/HCOM 标记识别 HCCL 通信。
        然后识别其他通信类别和常见计算类别。无法确定的事件保留为 other，
        不根据时间长短或事件顺序进行猜测。
    """
    name = event.get("name", "")
    category = event.get("cat", "")
    text = f"{name} {category}".lower()
    category_text = category.lower()

    if "hccl" in text or "hcom" in text:
        return {"kind": "communication", "communication_type": "hccl"}

    if "communication" in category_text or "comm" in category_text:
        return {"kind": "communication", "communication_type": "unknown"}

    compute_markers = ("kernel", "operator", "compute", "cpu_op", "gpu_op", "npu")
    for marker in compute_markers:
        if marker in category_text:
            return {"kind": "compute", "communication_type": ""}

    return {"kind": "other", "communication_type": ""}


def extract_events(trace):
    """把原始 trace 事件转换成统一的事件记录。

    输入：
        trace：由 load_trace 返回的 trace 字典，必须包含 traceEvents 列表。

    输出：
        返回事件记录列表。每条记录包含来源位置、名称、类别、阶段、事件类型、
        通信类型、开始时间、持续时间、进程号、线程号、事件/flow 标识和原始 args。
        时间字段沿用 Ascend trace 的微秒单位；缺失或类型不正确时使用 None，
        持续时间缺失时使用 0。

    处理内容：
        逐个读取原始事件，调用 classify_event 识别计算和通信事件，
        同时复制后续建立依赖可能需要的时间、进程、线程和 args 信息。
        本函数不会修改输入 trace，也不会推断事件之间的依赖关系。
    """
    normalized_events = []

    for index, event in enumerate(trace["traceEvents"]):
        name = event.get("name", "<unnamed>")
        if not isinstance(name, str):
            name = str(name)

        category = event.get("cat", "")
        phase = event.get("ph", "")
        event_args = event.get("args", {})
        if not isinstance(event_args, dict):
            event_args = {}

        start_time = event.get("ts")
        if not isinstance(start_time, (int, float)) or isinstance(start_time, bool):
            start_time = None

        duration = event.get("dur", 0)
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            duration = 0

        stream_id = event.get("stream")
        if stream_id is None:
            for stream_field in ("stream", "stream_id", "Stream", "Stream Id"):
                if stream_field in event_args:
                    stream_id = event_args[stream_field]
                    break
        if not isinstance(stream_id, (str, int, float)) or isinstance(stream_id, bool):
            stream_id = None

        classification = classify_event(event)
        normalized_events.append(
            {
                "source_index": index,
                "name": name,
                "category": category,
                "phase": phase,
                "kind": classification["kind"],
                "communication_type": classification["communication_type"],
                "start_time_micros": start_time,
                "duration_micros": duration,
                "process_id": event.get("pid"),
                "thread_id": event.get("tid"),
                "stream_id": stream_id,
                "event_id": event.get("id"),
                "event_id2": event.get("id2"),
                "bind_id": event.get("bind_id"),
                "scope": event.get("scope"),
                "flow_phase": event.get("bp"),
                "flow_scope": event.get("s"),
                "args": event_args,
            }
        )

    return normalized_events


def add_dependency(event, source_node_id, reason):
    """向事件中加入一条依赖边。

    输入：
        event：要增加依赖的目标事件记录。
        source_node_id：依赖来源事件的节点编号，使用从 1 开始的编号。
        reason：建立依赖的原因，例如 same_stream 或 correlation。

    输出：
        无返回值。函数会直接更新 event 中的 data_dependencies 和 dependency_reasons。

    处理内容：
        避免事件依赖自己，并避免重复加入相同的依赖边。
        如果同一条边由多个规则得到，则把多个原因合并保存。
    """
    target_node_id = event["node_id"]
    if source_node_id == target_node_id:
        return

    dependencies = event["data_dependencies"]
    reasons = event["dependency_reasons"]
    if source_node_id not in dependencies:
        dependencies.append(source_node_id)
        reasons[str(source_node_id)] = reason
    else:
        reason_key = str(source_node_id)
        if reason not in reasons[reason_key].split(","):
            reasons[reason_key] = f"{reasons[reason_key]},{reason}"


def sort_events_for_dependency(events):
    """按 trace 时间为依赖分析准备事件顺序。

    输入：
        events：extract_events 返回的统一事件记录列表。

    输出：
        返回按开始时间和原始位置排序的新列表，不修改输入列表。

    处理内容：
        有时间戳的事件排在前面，并按开始时间排序；时间戳相同时使用原始位置。
        没有时间戳的事件放在最后，避免用不完整时间信息建立错误顺序。
    """

    def sort_key(event):
        start_time = event["start_time_micros"]
        if start_time is None:
            return (1, 0, event["source_index"])
        return (0, start_time, event["source_index"])

    return sorted(events, key=sort_key)


def add_group_dependencies(events, group_key_name, reason):
    """为同一关联键的事件按顺序添加依赖。

    输入：
        events：统一事件记录列表。
        group_key_name：用于分组的字段名称，例如 correlation_key 或 stream_key。
        reason：依赖原因名称。

    输出：
        返回实际新增的依赖边数量。

    处理内容：
        将具有相同关联键的事件分组，按时间排序后连接相邻事件。
        只有共享明确关联键的事件才会建立依赖，不跨组推断依赖。
    """
    groups = {}
    for event in events:
        group_key = event.get(group_key_name)
        if group_key is None or event["start_time_micros"] is None:
            continue
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(event)

    edge_count = 0
    for group_events in groups.values():
        ordered_events = sort_events_for_dependency(group_events)
        for index in range(1, len(ordered_events)):
            target = ordered_events[index]
            source = ordered_events[index - 1]
            old_count = len(target["data_dependencies"])
            add_dependency(target, source["node_id"], reason)
            if len(target["data_dependencies"]) > old_count:
                edge_count += 1

    return edge_count


def build_dependencies(events):
    """根据明确关联信息建立保守的执行依赖。

    输入：
        events：extract_events 返回的统一事件记录列表。

    输出：
        返回一个二元组：
        - 带有 data_dependencies 和 dependency_reasons 的事件记录列表；
        - 依赖分析警告列表。

    处理内容：
        1. 复制事件记录，避免修改调用方传入的数据。
        2. 对共享 correlation、flow 或其他显式关联标识的事件建立依赖。
        3. 对同一进程、同一设备流中的计算和通信事件按时间顺序建立依赖。
        4. 不跨设备流根据时间先后建立依赖；缺少时间或流信息时记录警告。

        这些依赖只表示从输入 trace 中能够确认的关系，不能替代完整的设备执行图。
    """
    dependency_events = []
    warnings = []
    missing_stream_count = 0
    missing_timestamp_count = 0

    for event in events:
        copied_event = dict(event)
        copied_event["data_dependencies"] = []
        copied_event["dependency_reasons"] = {}
        dependency_events.append(copied_event)

    for event in dependency_events:
        source_index = event["source_index"]
        event["node_id"] = source_index + 1

        args = event["args"]
        correlation = None
        for field in ("correlation", "correlation_id", "External id", "external_id", "Record function id", "rf_id"):
            if field in args:
                correlation = args[field]
                break
        if correlation is not None:
            event["correlation_key"] = ("correlation", str(correlation))

        flow_id = event["event_id"]
        if flow_id is None:
            flow_id = event["bind_id"]
        if flow_id is not None and event["phase"] in ("s", "t", "f"):
            event["flow_key"] = ("flow", str(flow_id))

        if event["process_id"] is not None and event["stream_id"] is not None:
            event["stream_key"] = (str(event["process_id"]), str(event["stream_id"]))
        elif event["kind"] in ("compute", "communication"):
            missing_stream_count += 1

        if event["start_time_micros"] is None and event["kind"] in ("compute", "communication"):
            missing_timestamp_count += 1

    edge_count = 0
    edge_count += add_group_dependencies(dependency_events, "correlation_key", "correlation")
    edge_count += add_group_dependencies(dependency_events, "flow_key", "flow")

    stream_events = []
    for event in dependency_events:
        if event["kind"] in ("compute", "communication") and event.get("stream_key") is not None:
            stream_events.append(event)
    edge_count += add_group_dependencies(stream_events, "stream_key", "same_stream")

    if missing_stream_count:
        warnings.append(
            f"{missing_stream_count} compute or communication events have no process or stream information; "
            "same-stream dependencies were skipped"
        )
    if missing_timestamp_count:
        warnings.append(
            f"{missing_timestamp_count} compute or communication events have no timestamp; "
            "time-based ordering was skipped"
        )

    return dependency_events, warnings


def summarize_trace(trace, dependency_events, dependency_warnings):
    """统计 trace 中的原始事件和依赖结果。

    输入：
        trace：由 load_trace 返回的 trace 字典，必须包含 traceEvents 列表。
        dependency_events：build_dependencies 返回的事件记录列表。
        dependency_warnings：build_dependencies 返回的警告列表。

    输出：
        返回统计字典，包含事件总数、按 cat 分类的数量、按 ph 分类的数量、
        按统一事件类型分类的数量，以及依赖边数量和依赖原因数量。

    处理内容：
        遍历所有事件统计类别和阶段，遍历统一事件统计类型，
        再遍历依赖结果累计依赖边数量和各原因出现次数。
    """
    categories = {}
    phases = {}
    event_kinds = {}
    dependency_reasons = {}

    for event in trace["traceEvents"]:
        category = event.get("cat", "<missing>")
        phase = event.get("ph", "<missing>")
        categories[category] = categories.get(category, 0) + 1
        phases[phase] = phases.get(phase, 0) + 1

    for event in dependency_events:
        kind = event["kind"]
        event_kinds[kind] = event_kinds.get(kind, 0) + 1

    dependency_edge_count = 0
    for event in dependency_events:
        dependency_edge_count += len(event["data_dependencies"])
        for reasons in event["dependency_reasons"].values():
            for reason in reasons.split(","):
                dependency_reasons[reason] = dependency_reasons.get(reason, 0) + 1

    return {
        "event_count": len(trace["traceEvents"]),
        "categories": categories,
        "phases": phases,
        "event_kinds": event_kinds,
        "dependency_edge_count": dependency_edge_count,
        "dependency_reasons": dependency_reasons,
        "dependency_warnings": dependency_warnings,
    }


# Chakra 官方 generator 和 converter 使用的 schema 版本，写出时保持固定。
CHAKRA_SCHEMA_VERSION = "1.0.0"


def load_chakra_schema():
    """导入 Chakra 官方 protobuf 定义。

    输入：
        无参数。

    输出：
        返回 chakra.schema.protobuf.et_def_pb2 模块，包含 Node、GlobalMetadata 等消息类型。

    处理内容：
        延迟导入 Chakra 包，使读取和统计功能在未安装 Chakra 时仍可使用。
        未安装时给出明确的安装提示，而不是抛出难懂的导入错误。
    """
    try:
        from chakra.schema.protobuf import et_def_pb2
    except ImportError as error:
        raise ImportError(
            "Chakra is required to export protobuf traces. "
            "Install it with: pip install git+https://github.com/mlcommons/chakra.git"
        ) from error

    return et_def_pb2


def get_chakra_node_type(et_def_pb2, event):
    """把统一事件类型映射为 Chakra 节点类型。

    输入：
        et_def_pb2：Chakra protobuf 模块。
        event：带依赖信息的事件记录。

    输出：
        返回 Chakra 的节点类型枚举值。

    处理内容：
        计算事件映射为 COMP_NODE；通信事件中名称包含 send 或 recv 的分别映射为
        COMM_SEND_NODE 或 COMM_RECV_NODE，其余映射为 COMM_COLL_NODE；
        其他事件映射为 METADATA_NODE，保留记录但不当作计算任务。
        这里只按名称判断方向，没有解析真实的 P2P 语义。
    """
    if event["kind"] == "compute":
        return et_def_pb2.COMP_NODE
    if event["kind"] == "communication":
        name = event["name"].lower()
        if "send" in name:
            return et_def_pb2.COMM_SEND_NODE
        if "recv" in name or "receive" in name:
            return et_def_pb2.COMM_RECV_NODE
        return et_def_pb2.COMM_COLL_NODE
    return et_def_pb2.METADATA_NODE


def get_collective_type(et_def_pb2, name):
    """根据通信事件名称判断 collective 类型。

    输入：
        et_def_pb2：Chakra protobuf 模块。
        name：通信事件名称。

    输出：
        返回 Chakra 的 collective 类型枚举值；无法识别时返回 None。

    处理内容：
        去掉名称中的下划线和连字符并转成小写，再按关键字匹配常见 HCCL collective。
        只做名称匹配，不根据耗时或调用顺序推断通信类型。
    """
    normalized_name = name.replace("_", "").replace("-", "").lower()
    collective_types = {
        "allreduce": et_def_pb2.ALL_REDUCE,
        "allgather": et_def_pb2.ALL_GATHER,
        "reducescatter": et_def_pb2.REDUCE_SCATTER,
        "alltoall": et_def_pb2.ALL_TO_ALL,
        "broadcast": et_def_pb2.BROADCAST,
    }

    for keyword, collective_type in collective_types.items():
        if keyword in normalized_name:
            return collective_type

    return None


def build_chakra_nodes(dependency_events, et_def_pb2):
    """把带依赖的事件记录转换成 Chakra 节点。

    输入：
        dependency_events：build_dependencies 返回的事件记录列表。
        et_def_pb2：Chakra protobuf 模块。

    输出：
        返回一个字典，包含 metadata（GlobalMetadata 消息）和 nodes（Node 列表）。

    处理内容：
        1. 只为带开始时间的事件创建节点；没有时间的事件无法表达时间线，予以跳过。
        2. 节点编号从 1 开始重新分配，与 Chakra 官方 generator 的习惯一致。
        3. 节点开始时间写成相对 trace 最早开始时间的微秒偏移，
           元数据中保存绝对起止时间，便于回溯原始 trace。
        4. 依赖边只保留未被跳过的节点，并转换成新的节点编号。
        5. 通信节点写入 communication_type 和能识别的 comm_type；
           消息大小等未知属性不写入，避免写入猜测值。
    """
    kept_events = []
    for event in dependency_events:
        if event["start_time_micros"] is None:
            continue
        kept_events.append(event)

    old_id_to_new_id = {}
    for index, event in enumerate(kept_events):
        old_id_to_new_id[event["node_id"]] = index + 1

    start_times = []
    end_times = []
    for event in kept_events:
        start_time = event["start_time_micros"]
        start_times.append(start_time)
        end_times.append(start_time + event["duration_micros"])

    base_time = min(start_times) if start_times else 0
    finish_time = max(end_times) if end_times else base_time

    nodes = []
    for event in kept_events:
        node = et_def_pb2.Node()
        node.id = old_id_to_new_id[event["node_id"]]
        node.name = event["name"]
        node.type = get_chakra_node_type(et_def_pb2, event)
        node.start_time_micros = int(event["start_time_micros"] - base_time)
        node.duration_micros = int(event["duration_micros"])

        for dependency in event["data_dependencies"]:
            new_id = old_id_to_new_id.get(dependency)
            if new_id is not None and new_id not in node.data_deps:
                node.data_deps.append(new_id)

        node.attr.append(et_def_pb2.AttributeProto(name="source_index", int64_val=event["source_index"]))
        node.attr.append(et_def_pb2.AttributeProto(name="original_category", string_val=str(event["category"])))
        if event["stream_id"] is not None:
            node.attr.append(et_def_pb2.AttributeProto(name="stream", int64_val=int(event["stream_id"])))
        if event["communication_type"]:
            node.attr.append(
                et_def_pb2.AttributeProto(name="communication_type", string_val=event["communication_type"])
            )
            collective_type = get_collective_type(et_def_pb2, event["name"])
            if collective_type is not None:
                node.attr.append(et_def_pb2.AttributeProto(name="comm_type", int64_val=collective_type))

        nodes.append(node)

    metadata = et_def_pb2.GlobalMetadata(
        version=CHAKRA_SCHEMA_VERSION,
        attr=[
            et_def_pb2.AttributeProto(name="schema", string_val="Ascend"),
            et_def_pb2.AttributeProto(name="start_ts", uint64_val=int(base_time)),
            et_def_pb2.AttributeProto(name="finish_ts", uint64_val=int(finish_time)),
        ],
    )

    return {"metadata": metadata, "nodes": nodes}


def encode_varint(value):
    """把一个非负整数编码成 protobuf 使用的 varint 字节。

    输入：
        value：需要编码的非负整数，表示后续消息的字节长度。

    输出：
        返回编码后的 bytes。

    处理内容：
        每次取低 7 位，若还有剩余高位则把最高位置 1 表示继续。
        这是 Chakra 官方 protolib 写文件时使用的长度前缀格式。
    """
    buffer = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            buffer.append(byte | 0x80)
        else:
            buffer.append(byte)
            break
    return bytes(buffer)


def write_chakra_trace(output_path, chakra_nodes):
    """把 Chakra 元数据和节点写入 protobuf 文件。

    输入：
        output_path：输出文件路径。
        chakra_nodes：build_chakra_nodes 返回的字典，包含 metadata 和 nodes。

    输出：
        返回实际写入的节点数量。

    处理内容：
        先写 GlobalMetadata，再按列表顺序写每个 Node。
        每条消息前写入 varint 长度前缀，与 Chakra 官方 encodeMessage 的格式一致。
        文件以二进制方式写入，父目录不存在时自动创建。
    """
    output_file = Path(output_path)
    if output_file.parent != Path(""):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    messages = [chakra_nodes["metadata"]] + chakra_nodes["nodes"]
    with output_file.open("wb") as trace_file:
        for message in messages:
            data = message.SerializeToString()
            trace_file.write(encode_varint(len(data)))
            trace_file.write(data)

    return len(chakra_nodes["nodes"])


def write_chakra_json(output_path, chakra_nodes):
    """把 Chakra 元数据和节点写成 JSON 文件。

    输入：
        output_path：输出文件路径。
        chakra_nodes：build_chakra_nodes 返回的字典，包含 metadata 和 nodes。

    输出：
        返回实际写入的节点数量。

    处理内容：
        使用 protobuf 自带的 MessageToDict 把每条 message 转成普通字典，
        整体写成 {"metadata": ..., "nodes": [...]} 形式的单个 JSON 文档。
        字段命名与 chakra_jsonizer 的输出一致，长整数以字符串表示。
        父目录不存在时自动创建。
    """
    from google.protobuf import json_format

    output_file = Path(output_path)
    if output_file.parent != Path(""):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "metadata": json_format.MessageToDict(chakra_nodes["metadata"]),
        "nodes": [json_format.MessageToDict(node) for node in chakra_nodes["nodes"]],
    }
    output_file.write_text(json.dumps(document, ensure_ascii=False, indent=2))

    return len(chakra_nodes["nodes"])


def export_chakra_trace(input_path, output_path, output_format="protobuf"):
    """从 Ascend trace 生成 Chakra protobuf 文件。

    输入：
        input_path：trace_view.json 文件路径。
        output_path：Chakra 输出文件路径。
        output_format：输出格式，"protobuf" 表示 Chakra 官方二进制格式，
            "json" 表示单个 JSON 文档。

    输出：
        返回统计字典，并额外包含节点数量、输出路径和依赖警告。

    处理内容：
        依次执行读取、事件提取、依赖构建、节点转换和文件写出。
        本函数负责串联整个离线流程，不修改输入 trace。
    """
    trace = load_trace(input_path)
    events = extract_events(trace)
    dependency_events, warnings = build_dependencies(events)
    et_def_pb2 = load_chakra_schema()
    chakra_nodes = build_chakra_nodes(dependency_events, et_def_pb2)
    if output_format == "json":
        node_count = write_chakra_json(output_path, chakra_nodes)
    else:
        node_count = write_chakra_trace(output_path, chakra_nodes)

    summary = summarize_trace(trace, dependency_events, warnings)
    summary["chakra_node_count"] = node_count
    summary["output_path"] = str(output_path)
    return summary


def main():
    """处理命令行参数，统计 trace 并按需导出 Chakra 文件。

    输入：
        从命令行读取以下参数：
        - --input：trace_view.json 文件路径，必填。
        - --output：Chakra 输出文件路径，可选；不指定时只统计不导出。
        - --format：输出格式，可选 protobuf（默认）或 json。

    输出：
        成功时向标准输出打印格式化后的 JSON 结果。
        输入错误或缺少 Chakra 依赖时向标准错误输出错误信息，并以退出码 2 结束程序。

    处理内容：
        读取 trace 并完成事件提取和依赖分析，再根据是否指定 --output
        决定只输出统计信息，还是同时写出 Chakra execution trace。
    """
    parser = argparse.ArgumentParser(
        description="Convert an Ascend trace_view.json into a Chakra execution trace. "
        "Without --output, only input statistics are reported."
    )
    parser.add_argument("--input", required=True, help="Path to an Ascend trace_view.json file.")
    parser.add_argument("--output", help="Path for the Chakra execution trace output file.")
    parser.add_argument(
        "--format",
        choices=["protobuf", "json"],
        default="protobuf",
        help="Output format: Chakra protobuf (default) or a single JSON document.",
    )
    args = parser.parse_args()

    try:
        if args.output:
            summary = export_chakra_trace(args.input, args.output, args.format)
        else:
            trace = load_trace(args.input)
            dependency_events, warnings = build_dependencies(extract_events(trace))
            summary = summarize_trace(trace, dependency_events, warnings)
    except (OSError, UnicodeError, ValueError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
