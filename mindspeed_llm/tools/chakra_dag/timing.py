# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""解析 Ascend profiling 产物，并把实测时长 join 到 DAG 上。

## 本模块只做"测量"，不参与"结构"

结构（谁依赖谁）来自 ``capture`` 的运行时观测。本模块的唯一职责是补上
"每个节点实际跑了多久"，以及在融合算子无法内部拆分时提供估算依据。

这条分工很重要：一旦用 trace 反推结构，就会引入一批无法确定的问题
（跨卡 collective 配对、跨流连边、计算与通信谁等谁）。

## 输入

- ``trace_view.json``：顶层事件数组（或含 ``traceEvents`` 的对象）。
- ``profiler_metadata.json``：提供 rank / world_size / 并行组翻译表。
- 可选 ``communication.json``：提供真实传输时间与带宽。

## 时长 join 的两种方式

1. **按名字匹配**（融合算子）：运行时记录的 ``comm_name`` 通常出现在
   ``hcom_<type>__<name>_<seq>`` 中段，可直接对号。
2. **按调用序号匹配**（普通算子）：同一 rank 上第 N 个同名 kernel 对应第 N 次调用。
   仅在名字匹配失败时使用，并在节点上标注 ``time_match=sequence``。

两种都失败时节点保留 ``timing.source = "unknown"``，不编造数字。
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import ir

#: 计算事件的判定：Process 名为 Ascend Hardware，且 Task Type 在下列集合内。
COMPUTE_TASK_TYPES = ("AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV")

#: 需要明确排除的 Task Type（微步骤、内存搬运、同步记录等）。
EXCLUDED_TASK_TYPES = (
    "EVENT_RECORD",
    "EVENT_WAIT",
    "SDMA",
    "SDMA_SQE",
    "PCIE_DMA_SQE",
    "Write Value",
    "Notify_Wait",
    "PROFILING_DISABLE",
)

#: 名字前缀排除项：通信微步骤、内存搬运。比较时统一转小写，
#: 因为 trace 里的大小写并不稳定（``Memcpy`` / ``memcpy`` 都出现过）。
EXCLUDED_NAME_PREFIXES = ("hcom", "memcpy", "notify", "reduce_inline")

#: 集合通信类型识别：按 hcom 名称里的片段映射。
COMM_NAME_TO_TYPE = (
    ("allreduce", "all_reduce"),
    ("reduce_scatter", "reduce_scatter"),
    ("reducescatter", "reduce_scatter"),
    ("allgather", "all_gather"),
    ("all_gather", "all_gather"),
    ("alltoall", "all_to_all"),
    ("all_to_all", "all_to_all"),
    ("broadcast", "broadcast"),
    ("barrier", "barrier"),
)


def _as_float(value: Any) -> Optional[float]:
    """trace 里的 ts/dur 有时是字符串，统一转成 float；失败返回 None。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_trace_events(path: str) -> List[Dict[str, Any]]:
    """读 trace 文件，返回事件列表。

    支持三种形态：顶层数组、``{"traceEvents": [...]}``、以及带 BOM 的对象文件。
    """
    with open(path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        events = payload.get("traceEvents")
        if isinstance(events, list):
            return events
    raise ValueError(f"无法识别的 trace 结构（既不是数组也没有 traceEvents）: {path}")


def load_profiler_metadata(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def build_process_name_map(events: Iterable[Dict[str, Any]]) -> Dict[int, str]:
    """由 ``ph=M`` 的 process_name 元数据建立 pid -> 进程名映射。

    trace 事件本身不带 rank/device 字段，进程名（``Ascend Hardware``、
    ``Communication``）是区分事件层级的唯一可靠依据。
    """
    mapping: Dict[int, str] = {}
    for event in events:
        if event.get("ph") != "M":
            continue
        if event.get("name") != "process_name":
            continue
        pid = event.get("pid")
        args = event.get("args") or {}
        name = args.get("name")
        if pid is not None and name is not None:
            mapping[pid] = str(name)
    return mapping


def resolve_rank(metadata: Dict[str, Any], trace_path: str) -> Optional[int]:
    """从 profiler_metadata.json 里取 rank。

    优先级：``distributed_args.rank`` > 目录名中的 ``rankN`` > 无。
    事件本身没有 rank 字段，所以只能靠元数据或目录约定。
    """
    distributed = metadata.get("distributed_args") or {}
    rank = distributed.get("rank")
    if rank is not None:
        try:
            return int(rank)
        except (TypeError, ValueError):
            pass
    parent = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(trace_path))))
    if parent.startswith("rank") and parent[4:].isdigit():
        return int(parent[4:])
    return None


def translate_group(metadata: Dict[str, Any], group_name: str) -> Optional[List[int]]:
    """把 rank-local 的 group_name 翻译成 global rank 列表。

    这是跨卡合并的前提：同一个逻辑组在不同 rank 上可能叫 ``group_name_17`` 或
    ``group_name_16``，只有 ``parallel_group_info`` 里的 ``global_ranks`` 是全局稳定的。
    """
    for key in ("parallel_group_info", "group_info"):
        table = metadata.get(key)
        if not isinstance(table, dict):
            continue
        for entry in table.values():
            if not isinstance(entry, dict):
                continue
            name = entry.get("group_name")
            if name != group_name:
                continue
            ranks = entry.get("global_ranks") or entry.get("group_global_rank")
            if isinstance(ranks, list):
                return [int(item) for item in ranks]
    return None


def extract_compute_timings(
    events: Sequence[Dict[str, Any]], base_ts_us: Optional[float] = None
) -> Dict[str, List[Tuple[float, float]]]:
    """抽取计算 kernel 的 ``(name, start_us, dur_us)``，按名字分组。

    只保留 ``Ascend Hardware`` 进程下的 ``ph=X`` 事件，且 Task Type 属于
    ``COMPUTE_TASK_TYPES``。排除项见 ``EXCLUDED_TASK_TYPES`` 与
    ``EXCLUDED_NAME_PREFIXES``。
    """
    process_names = build_process_name_map(events)
    result: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        pid = event.get("pid")
        if process_names.get(pid) != "Ascend Hardware":
            continue
        args = event.get("args") or {}
        task_type = args.get("Task Type")
        if task_type is None:
            continue
        if str(task_type) in EXCLUDED_TASK_TYPES:
            continue
        if str(task_type) not in COMPUTE_TASK_TYPES:
            continue
        name = str(event.get("name", ""))
        lowered_name = name.lower()
        if any(lowered_name.startswith(prefix) for prefix in EXCLUDED_NAME_PREFIXES):
            continue
        start = _as_float(event.get("ts"))
        dur = _as_float(event.get("dur"))
        if start is None or dur is None:
            continue
        if base_ts_us is not None:
            start -= base_ts_us
        result[name].append((start, dur))
    for item in result.values():
        item.sort(key=lambda pair: pair[0])
    return dict(result)


def extract_collective_timings(
    events: Sequence[Dict[str, Any]], base_ts_us: Optional[float] = None
) -> List[Dict[str, Any]]:
    """抽取集合通信宏事件。

    判定：``process_name=Communication``、``ph=X``、名字以 ``hcom`` 或 ``host`` 开头，
    且不是 send/receive（P2P）也不是微步骤。
    """
    process_names = build_process_name_map(events)
    result: List[Dict[str, Any]] = []
    for event in events:
        if event.get("ph") != "X":
            continue
        pid = event.get("pid")
        if process_names.get(pid) != "Communication":
            continue
        name = str(event.get("name", ""))
        lowered = name.lower()
        if not (lowered.startswith("hcom") or lowered.startswith("host")):
            continue
        if "send" in lowered or "recv" in lowered:
            continue
        if any(marker in lowered for marker in ("memcpy", "reduce_inline", "notify_")):
            continue
        start = _as_float(event.get("ts"))
        dur = _as_float(event.get("dur"))
        if start is None or dur is None:
            continue
        if base_ts_us is not None:
            start -= base_ts_us
        args = event.get("args") or {}
        record = {
            "name": name,
            "start_us": start,
            "duration_us": dur,
            "comm_type": classify_comm_name(name),
            "count": args.get("count"),
            "dtype": args.get("data_type"),
            "group_name": args.get("group_name") or "",
            "raw_name": name,
        }
        width = ir.dtype_size(record["dtype"])
        if isinstance(record["count"], (int, float)) and width:
            record["nbytes"] = int(record["count"]) * width
        result.append(record)
    result.sort(key=lambda item: item["start_us"])
    return result


def classify_comm_name(name: str) -> str:
    """由 hcom 名称推断 collective 类型；识别失败返回空串。"""
    normalized = name.lower().replace("-", "_")
    for keyword, comm_type in COMM_NAME_TO_TYPE:
        if keyword in normalized:
            return comm_type
    return ""


def load_communication_summary(path: str) -> List[Dict[str, Any]]:
    """读 ``communication.json``，取真实传输时间与带宽。

    该文件里的 ``Elapsed Time`` 包含等待，``Transit Time`` 才是真实传输；
    两者都要保留，否则无法区分"通信慢"和"等待久"。
    """
    if not os.path.exists(path):
        return []
    try:
        payload = load_profiler_metadata(path)
    except (ValueError, OSError):
        return []
    records: List[Dict[str, Any]] = []

    def walk(node, depth=0):
        if depth > 4:
            return
        if isinstance(node, dict):
            lowered = {str(key).lower(): value for key, value in node.items()}
            if any("transit" in key for key in lowered):
                records.append(
                    {
                        "transit_size_mb": _first_number(lowered, ("transit size", "transit_size")),
                        "transit_time_ns": _first_number(lowered, ("transit time", "transit_time")),
                        "bandwidth_gbps": _first_number(lowered, ("bandwidth",)),
                        "elapsed_time_ns": _first_number(lowered, ("elapsed time", "elapsed_time")),
                        "group_name": str(node.get("Group Name") or node.get("group_name") or ""),
                        "comm_type": str(node.get("Communication Type") or node.get("comm_type") or ""),
                    }
                )
                return
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                walk(value, depth + 1)

    walk(payload)
    return records


def _first_number(mapping: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        for actual, value in mapping.items():
            if actual.startswith(key):
                number = _as_float(value)
                if number is not None:
                    return number
    return None


def match_collective_measured(node: ir.Node, candidates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """为一个 collective 节点找最可能的实测记录。

    匹配依据（按强度）：
        1. ``comm_name`` 出现在事件名里；
        2. 组名 + 类型一致；
        3. 都没有时返回 None，留给调用方按序号匹配。
    """
    comm = node.comm
    if comm is None:
        return None
    if comm.comm_name:
        for record in candidates:
            if comm.comm_name in record["name"]:
                return record
    for record in candidates:
        if not comm.comm_type or record["comm_type"] != comm.comm_type:
            continue
        if comm.group_name and record["group_name"] != comm.group_name:
            continue
        if comm.nbytes is not None and record.get("nbytes") is not None:
            if int(record["nbytes"]) != int(comm.nbytes):
                continue
        return record
    return None


def apply_compute_timings(
    graph: ir.RankGraph, compute_map: Dict[str, List[Tuple[float, float]]], match_sequence: bool = True
) -> Dict[str, int]:
    """把计算的实测时长贴到计算节点上。

    同名 kernel 可能在一层/多步内出现多次，所以按序消费：第 N 个同名节点取第 N 个
    同名测点。测点不足时该节点保持 unknown，绝不重复使用同一测点。
    """
    cursor: Dict[str, int] = defaultdict(int)
    stats = {"measured": 0, "unknown": 0}
    for node in graph.nodes:
        if node.kind != ir.COMP:
            continue
        if node.attrs.get("is_sync_marker"):
            continue
        series = compute_map.get(node.name)
        if not series:
            # 名字不完全一致时做一次前缀匹配（框架会在名字上加后缀）。
            for candidate, values in compute_map.items():
                if candidate.startswith(node.name) or node.name.startswith(candidate):
                    series = values
                    break
        if not series:
            stats["unknown"] += 1
            continue
        index = cursor[node.name]
        if index >= len(series):
            if not match_sequence:
                stats["unknown"] += 1
                continue
            index = len(series) - 1
        start, dur = series[index]
        cursor[node.name] = index + 1
        node.timing = ir.Timing(start_us=start, duration_us=dur, source=ir.TIME_MEASURED)
        stats["measured"] += 1
    return stats


def apply_collective_timings(
    graph: ir.RankGraph,
    candidates: Sequence[Dict[str, Any]],
    used: Optional[set] = None,
) -> Dict[str, int]:
    """把通信的实测时长贴到 collective 节点上。"""
    used = used if used is not None else set()
    stats = {"measured": 0, "unknown": 0}
    for node in graph.nodes:
        if node.kind != ir.COMM_COLL:
            continue
        match = None
        for index, record in enumerate(candidates):
            if index in used:
                continue
            candidate = match_collective_measured(node, [record])
            if candidate is not None:
                match = (index, record)
                break
        if match is None:
            stats["unknown"] += 1
            continue
        index, record = match
        used.add(index)
        node.timing = ir.Timing(
            start_us=record["start_us"],
            duration_us=record["duration_us"],
            source=ir.TIME_MEASURED,
            comm_time_us=record["duration_us"],
        )
        stats["measured"] += 1
    return stats


def apply_fused_split(
    graph: ir.RankGraph,
    mode: str = ir.SPLIT_ESTIMATE,
    default_bandwidth_gbps: Optional[float] = None,
) -> Dict[str, int]:
    """把融合算子的实测总时长拆给内部各节点。

    融合体的每个节点共享同一个实测包络（``fused_group``），本函数按 ``mode`` 决定
    如何把包络分给通信段和计算段：

    - ``split-measured``：单个节点已有独立实测值时直接用（说明 trace 里拆开了）。
    - ``split-estimate``：用 ``bytes / bandwidth`` 估通信时间，余量给计算。
    - ``merged``：不拆分，通信节点用整个包络，并标注 ``time_split=merged``。

    返回值是各模式的命中计数，写进报告，便于判断图里有多少是估算。
    """
    stats: Dict[str, int] = {"groups": 0, "split_measured": 0, "split_estimated": 0, "merged": 0}
    groups: Dict[str, List[ir.Node]] = defaultdict(list)
    for node in graph.nodes:
        group = node.attrs.get("fused_group")
        if group:
            groups[group].append(node)

    for group_id, members in groups.items():
        stats["groups"] += 1
        known = [node for node in members if node.timing.duration_us is not None]
        if not known:
            for node in members:
                node.attrs["time_split"] = ir.TIME_UNKNOWN
            continue
        envelope = max(node.timing.duration_us for node in known)
        start = min((node.timing.start_us for node in known if node.timing.start_us is not None), default=0.0)
        comm_node = next((node for node in members if node.kind == ir.COMM_COLL), None)
        comp_nodes = [node for node in members if node.kind == ir.COMP]

        if mode == ir.SPLIT_MERGED or comm_node is None or not comp_nodes:
            for node in members:
                node.timing.start_us = start
                node.timing.duration_us = envelope
                node.timing.source = ir.TIME_MEASURED
                node.attrs["time_split"] = ir.SPLIT_MERGED
            stats["merged"] += 1
            continue

        # 已经分别测到过（split-measured 的充分条件）：通信节点有自己的实测时长，
        # 且与包络不同源。此时不再覆盖。
        comm_measured = comm_node.timing.duration_us is not None and comm_node.timing.source == ir.TIME_MEASURED
        has_independent = any(node.attrs.get("time_independent") for node in members)
        if mode == ir.SPLIT_MEASURED and comm_measured and has_independent:
            for node in members:
                node.attrs["time_split"] = ir.SPLIT_MEASURED
            stats["split_measured"] += 1
            continue

        nbytes = comm_node.comm.nbytes if comm_node.comm else None
        bandwidth = default_bandwidth_gbps or comm_node.timing.bandwidth_gbps
        comm_time: Optional[float] = None
        if mode in (ir.SPLIT_MEASURED, ir.SPLIT_ESTIMATE) and nbytes and bandwidth:
            # bytes / (GB/s) -> 微秒：bytes / 1e9 GB * 1e6 us
            comm_time = (float(nbytes) / 1e9) / float(bandwidth) * 1e6

        if comm_time is None or comm_time > envelope:
            # 估不出或估计值超过包络：退化为共享包络，并明确标注。
            for node in members:
                node.timing.start_us = start
                node.timing.duration_us = envelope
                node.timing.source = ir.TIME_ESTIMATED
                node.attrs["time_split"] = ir.SPLIT_MERGED
            stats["merged"] += 1
            continue

        order = comm_node.attrs.get("fused_order", "comm_first")
        if order == "comm_first":
            comm_node.timing.start_us = start
            comp_start = start + comm_time
        else:
            comm_node.timing.start_us = start + (envelope - comm_time)
            comp_start = start
        comm_node.timing.duration_us = comm_time
        comm_node.timing.source = ir.TIME_ESTIMATED
        comm_node.timing.comm_time_us = comm_time
        comm_node.timing.bandwidth_gbps = bandwidth
        comm_node.attrs["time_split"] = ir.SPLIT_ESTIMATE

        comp_duration = max(envelope - comm_time, 0.0) if order == "comm_first" else envelope
        for node in comp_nodes:
            node.timing.start_us = comp_start
            node.timing.duration_us = comp_duration if order == "comm_first" else max(envelope, comm_time)
            node.timing.source = ir.TIME_ESTIMATED
            node.attrs["time_split"] = ir.SPLIT_ESTIMATE
        stats["split_estimated"] += 1
    return stats


def normalize_timeline(graph: ir.RankGraph) -> None:
    """把所有时间平移到从 0 开始。

    Chakra 的 ``start_time_micros`` 是 uint64，且下游仿真器按相对时间处理；
    昇腾 trace 的 ``ts`` 是 Unix epoch 微秒（约 1.7e15），直接写入会让仿真器
    把整段时间视为发生在遥远的未来。绝对基准保存在 ``manifest.json``。
    """
    starts = [node.timing.start_us for node in graph.nodes if node.timing.start_us is not None]
    if not starts:
        return
    offset = min(starts)
    if offset == 0:
        return
    for node in graph.nodes:
        if node.timing.start_us is not None:
            node.timing.start_us -= offset
    graph.meta["time_offset_us"] = offset
