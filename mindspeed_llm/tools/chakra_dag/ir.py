# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""通算 DAG 的中间表示（IR）。

本模块只依赖 Python 标准库，是运行期采集、离线 trace 补全、protobuf 写出三者之间
的公共契约。设计目标有两个：

1. **结构是一等公民**：依赖关系来自运行时观测，不是从时间戳反推。每条边都带
   ``source`` 字段记录它是怎么来的，便于审计。
2. **可跨进程传输**：所有对象都能无损地转成 JSON（``to_dict`` / ``from_dict``），
   因为采集发生在训练进程内，合并与写出发生在导出子进程里。

术语：

- ``Event``：采集器直接产出的原始记录，一条对应一次算子调用或一次通信调用。
- ``Node``：进入 DAG 的节点。一个 ``Event`` 可能展开成多个 ``Node``
  （融合算子），也可能不产生任何节点（被过滤的辅助调用）。
- ``Edge``：不单独存对象，而是以 ``Node.preds`` 的形式记录前驱节点 id。
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 节点种类：与 Chakra NodeType 一一对应，writer 负责翻译。
# 用字符串而不用 IntEnum，是为了让 JSON 中间文件可读、可 grep。
# ---------------------------------------------------------------------------
COMP = "comp"
COMM_COLL = "comm_coll"
COMM_SEND = "comm_send"
COMM_RECV = "comm_recv"
METADATA = "metadata"

NODE_KINDS = (COMP, COMM_COLL, COMM_SEND, COMM_RECV, METADATA)

# 事件种类：比节点种类多一个 fused，表示"需要展开的融合算子"。
EVENT_COMPUTE = "compute"
EVENT_COMM_COLL = "comm_coll"
EVENT_P2P_SEND = "p2p_send"
EVENT_P2P_RECV = "p2p_recv"
EVENT_FUSED = "fused"
EVENT_KINDS = (EVENT_COMPUTE, EVENT_COMM_COLL, EVENT_P2P_SEND, EVENT_P2P_RECV, EVENT_FUSED)

# 依赖来源：写进每条边的 source 字段，用于回答"这条边凭什么存在"。
DEP_TENSOR = "tensor"  # 张量版本号/就地写观测到的真实数据依赖
DEP_SEQUENCE = "sequence"  # 同一 Python 调用线程内的先后顺序
DEP_COLLECTIVE = "collective"  # 集合通信的入队/完成配对
DEP_P2P = "p2p"  # 点对点 send/recv 配对
DEP_FUSED = "fused"  # 融合算子展开时内部 comm 与 comp 的共同前驱/后继
DEP_INFERRED = "inferred"  # 没有任何证据、按时间窗兜底推断（默认不启用）
DEP_ROOT = "root"  # 图入口

# 时长来源：区分"测出来的"和"算出来的"，避免把估算当精确值。
TIME_MEASURED = "measured"
TIME_ESTIMATED = "estimated"
TIME_UNKNOWN = "unknown"

# 融合算子时长拆分策略，含义见 ``fused_ops`` 模块文档。
SPLIT_MEASURED = "split-measured"
SPLIT_ESTIMATE = "split-estimate"
SPLIT_MERGED = "merged"

# 集合通信类型：与 Chakra CollectiveCommType 名称一致（大写形式见 writer）。
COMM_TYPES = (
    "all_reduce",
    "reduce",
    "all_gather",
    "gather",
    "scatter",
    "broadcast",
    "all_to_all",
    "reduce_scatter",
    "reduce_scatter_block",
    "barrier",
)

# 常见 dtype 的字节宽度，用于在缺少显式 size 时换算通信量。
DTYPE_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "fp8": 1,
    "float8": 1,
    "int16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int32": 4,
    "int": 4,
    "fp32": 4,
    "float": 4,
    "float32": 4,
    "int64": 8,
    "long": 8,
    "fp64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


def dtype_size(dtype: Optional[str]) -> Optional[int]:
    """返回 dtype 的字节宽度；无法识别时返回 None，调用方据此决定是否换算失败。"""
    if not dtype:
        return None
    return DTYPE_SIZES.get(str(dtype).strip().lower())


@dataclass
class Timing:
    """节点的时长信息。

    ``start_us`` / ``duration_us`` 都是相对该 rank 采集窗口起点的微秒数，避免把
    Unix epoch 绝对时间戳写进 Chakra 节点（``start_time_micros`` 是 uint64 且
    下游仿真器按相对时间处理）。绝对时间基准保存在 ``manifest.json``。
    """

    start_us: Optional[float] = None
    duration_us: Optional[float] = None
    source: str = TIME_UNKNOWN
    # 通信节点的细分：真实传输时间可能与整体耗时不同（含等待）。
    comm_time_us: Optional[float] = None
    bandwidth_gbps: Optional[float] = None

    def is_known(self) -> bool:
        return self.start_us is not None and self.duration_us is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_us": self.start_us,
            "duration_us": self.duration_us,
            "source": self.source,
            "comm_time_us": self.comm_time_us,
            "bandwidth_gbps": self.bandwidth_gbps,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Timing":
        if not data:
            return cls()
        return cls(
            start_us=data.get("start_us"),
            duration_us=data.get("duration_us"),
            source=data.get("source", TIME_UNKNOWN),
            comm_time_us=data.get("comm_time_us"),
            bandwidth_gbps=data.get("bandwidth_gbps"),
        )


@dataclass
class CommMeta:
    """一次通信调用的语义信息。

    这些字段全部来自运行时观测（调用参数、process group 对象、张量 shape），
    不依赖 trace 名称解析，因此比事后从 ``hcom_*`` 事件名推断可靠。
    """

    comm_type: str = ""
    group_name: str = ""
    # HCCL 通信域名字，通常与 profiling 里的 ``hcom_<type>__<name>_<seq>`` 中段一致，
    # 是把运行时记录与 trace 实测时长对上号的关键。为空表示不可 join。
    comm_name: str = ""
    count: Optional[int] = None
    dtype: Optional[str] = None
    nbytes: Optional[int] = None
    # P2P 方向信息。
    src_rank: Optional[int] = None
    dst_rank: Optional[int] = None
    tag: Optional[int] = None
    # 逻辑组翻译后的 global rank 列表；None 表示未解析。
    participants: Optional[List[int]] = None
    # 同 (group, type, count, dtype) 下的第几次出现，SPMD 下各 rank 一致，
    # 是跨卡合并 collective 节点的确定性主键。
    occurrence: int = 0
    # 该通信是否与另一段算子融合（由 fused_ops 展开时填充）。
    fused_with: Optional[str] = None
    # 融合展开时这一段是通信还是计算。
    fused_role: Optional[str] = None

    def match_key(self) -> Optional[Tuple]:
        """跨卡合并用的匹配键。

        缺少 group/type/count 任一项时返回 None，表示这个 collective 无法安全地
        跨卡配对，调用方应保留为 rank-local 节点并在报告里标注。
        """
        if not self.comm_type or not self.group_name:
            return None
        if self.nbytes is None and self.count is None:
            return None
        size = self.nbytes if self.nbytes is not None else self.count
        return (self.group_name, self.comm_type, size, self.dtype, self.occurrence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comm_type": self.comm_type,
            "group_name": self.group_name,
            "comm_name": self.comm_name,
            "count": self.count,
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "src_rank": self.src_rank,
            "dst_rank": self.dst_rank,
            "tag": self.tag,
            "participants": self.participants,
            "occurrence": self.occurrence,
            "fused_with": self.fused_with,
            "fused_role": self.fused_role,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CommMeta":
        if not data:
            return cls()
        return cls(
            comm_type=data.get("comm_type", ""),
            group_name=data.get("group_name", ""),
            comm_name=data.get("comm_name", ""),
            count=data.get("count"),
            dtype=data.get("dtype"),
            nbytes=data.get("nbytes"),
            src_rank=data.get("src_rank"),
            dst_rank=data.get("dst_rank"),
            tag=data.get("tag"),
            participants=data.get("participants"),
            occurrence=data.get("occurrence", 0),
            fused_with=data.get("fused_with"),
            fused_role=data.get("fused_role"),
        )


@dataclass
class FusedSpec:
    """融合算子的展开说明。

    ``ops`` 按执行顺序列出融合在一起的算子，例如 ``["all_gather", "matmul"]``。
    展开后的节点在图中是**兄弟关系**（共同前驱、共同后继），而不是串行链——
    因为 MC2 这类融合的意义就在于通信与计算重叠，串行展开会强加一条不存在的依赖。
    """

    ops: List[str] = field(default_factory=list)
    # 融合体里通信部分对应的 collective 类型；为空表示这段融合没有集合通信。
    comm_type: str = ""
    group_name: str = ""
    comm_name: str = ""
    count: Optional[int] = None
    dtype: Optional[str] = None
    nbytes: Optional[int] = None
    participants: Optional[List[int]] = None
    occurrence: int = 0
    # "comm_first" / "comm_last"：融合体内通信相对计算的位置，用于估算时间切分。
    order: str = "comm_first"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ops": list(self.ops),
            "comm_type": self.comm_type,
            "group_name": self.group_name,
            "comm_name": self.comm_name,
            "count": self.count,
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "participants": self.participants,
            "occurrence": self.occurrence,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["FusedSpec"]:
        if not data:
            return None
        return cls(
            ops=list(data.get("ops", [])),
            comm_type=data.get("comm_type", ""),
            group_name=data.get("group_name", ""),
            comm_name=data.get("comm_name", ""),
            count=data.get("count"),
            dtype=data.get("dtype"),
            nbytes=data.get("nbytes"),
            participants=data.get("participants"),
            occurrence=data.get("occurrence", 0),
            order=data.get("order", "comm_first"),
        )


@dataclass
class Event:
    """采集器直接产出的记录。

    ``deps`` 指向同一 rank 内更早的 event id，表示"等到这些事件完成后本事件才开始"。
    对普通算子，依赖来自张量版本号观测；对通信，依赖来自入队与 ``wait()``。
    """

    eid: str
    seq: int
    kind: str
    name: str
    deps: List[str] = field(default_factory=list)
    dep_sources: List[str] = field(default_factory=list)
    comm: Optional[CommMeta] = None
    fused: Optional[FusedSpec] = None
    # 便于排查的自由字段：shape、dtype、调用点等。
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind: {self.kind}")
        if len(self.deps) != len(self.dep_sources):
            # 允许不填 dep_sources，此时统一视为序列依赖。
            self.dep_sources = self.dep_sources or [DEP_SEQUENCE] * len(self.deps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eid": self.eid,
            "seq": self.seq,
            "kind": self.kind,
            "name": self.name,
            "deps": list(self.deps),
            "dep_sources": list(self.dep_sources),
            "comm": self.comm.to_dict() if self.comm else None,
            "fused": self.fused.to_dict() if self.fused else None,
            "attrs": dict(self.attrs),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(
            eid=data["eid"],
            seq=data["seq"],
            kind=data["kind"],
            name=data["name"],
            deps=list(data.get("deps", [])),
            dep_sources=list(data.get("dep_sources", [])),
            comm=CommMeta.from_dict(data.get("comm")),
            fused=FusedSpec.from_dict(data.get("fused")),
            attrs=dict(data.get("attrs", {})),
        )


@dataclass
class Node:
    """进入 DAG 的节点。

    ``id`` 在整张合并图内唯一，格式为 ``r<rank>:<local_id>``，跨卡合并 collective
    时由发起合并的一方统一改写所有参与 rank 的节点 id。
    """

    id: str
    kind: str
    name: str
    rank: int
    seq: int
    preds: List[str] = field(default_factory=list)
    pred_sources: Dict[str, str] = field(default_factory=dict)
    succs: List[str] = field(default_factory=list)
    comm: Optional[CommMeta] = None
    timing: Timing = field(default_factory=Timing)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind: {self.kind}")

    def add_pred(self, node_id: str, source: str) -> bool:
        """追加一条前驱边，返回是否真的新增（去重）。"""
        if node_id == self.id:
            return False
        if node_id in self.pred_sources:
            return False
        self.preds.append(node_id)
        self.pred_sources[node_id] = source
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "rank": self.rank,
            "seq": self.seq,
            "preds": list(self.preds),
            "pred_sources": dict(self.pred_sources),
            "succs": list(self.succs),
            "comm": self.comm.to_dict() if self.comm else None,
            "timing": self.timing.to_dict(),
            "attrs": dict(self.attrs),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        return cls(
            id=data["id"],
            kind=data["kind"],
            name=data["name"],
            rank=data.get("rank", 0),
            seq=data.get("seq", 0),
            preds=list(data.get("preds", [])),
            pred_sources=dict(data.get("pred_sources", {})),
            succs=list(data.get("succs", [])),
            comm=CommMeta.from_dict(data.get("comm")),
            timing=Timing.from_dict(data.get("timing")),
            attrs=dict(data.get("attrs", {})),
        )


@dataclass
class RankGraph:
    """单个 rank 的图，附带该 rank 的元数据。"""

    rank: int
    nodes: List[Node] = field(default_factory=list)
    # 采集窗口的绝对起点（Unix epoch 微秒），用于和 trace 对齐。
    base_ts_us: Optional[float] = None
    # device_id → rank 映射等采集期信息。
    meta: Dict[str, Any] = field(default_factory=dict)

    def node_map(self) -> Dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def index_succs(self) -> None:
        """由 preds 反推 succs，保证两者一致。"""
        for node in self.nodes:
            node.succs = []
        by_id = self.node_map()
        for node in self.nodes:
            for pred_id in node.preds:
                pred = by_id.get(pred_id)
                if pred is not None:
                    pred.succs.append(node.id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "base_ts_us": self.base_ts_us,
            "meta": dict(self.meta),
            "nodes": [node.to_dict() for node in self.nodes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RankGraph":
        return cls(
            rank=data.get("rank", 0),
            nodes=[Node.from_dict(item) for item in data.get("nodes", [])],
            base_ts_us=data.get("base_ts_us"),
            meta=dict(data.get("meta", {})),
        )


def dump_rank_graph(graph: RankGraph, path: str) -> None:
    """把一个 rank 的图写成 JSON 中间文件。

    为什么需要中间文件：采集在训练进程内完成，写出在独立子进程完成，
    跨进程只能传文件或管道。JSON 也便于现场排查。
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(graph.to_dict(), handle, ensure_ascii=False, indent=1)


def load_rank_graph(path: str) -> RankGraph:
    with open(path, "r", encoding="utf-8") as handle:
        return RankGraph.from_dict(json.load(handle))


def dump_events(events: Sequence[Event], path: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """把事件序列写成 JSON。

    自动创建父目录：采集发生在训练进程里，调用方不该为了写一个分析产物去关心目录
    是否存在（尤其默认输出目录是 ``<profile_save_path>/chakra_dag``，profiler 不会建它）。
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "meta": dict(meta or {}),
        "events": [event.to_dict() for event in events],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)


def load_events(path: str) -> Tuple[List[Event], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = [Event.from_dict(item) for item in payload.get("events", [])]
    return events, dict(payload.get("meta", {}))
