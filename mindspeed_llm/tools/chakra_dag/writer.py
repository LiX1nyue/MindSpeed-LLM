# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""写出 Chakra execution trace（protobuf）与合并图 JSON。

## 依赖策略

``chakra`` 官方包不在 PyPI（``pip install chakra`` 只会装到一个 2010 年的 Python2
包），正确装法是::

    pip install git+https://github.com/mlcommons/chakra.git

它的 ``pyproject.toml`` 会拉取 ``HolisticTraceAnalysis`` 和 ``grpcio-tools`` 来生成
``_pb2`` 模块，在离线昇腾节点上可能装不上。因此本模块：

- **顶层不 import chakra**，离线合并、单元测试、JSON 输出都不需要它；
- 只在真正写 protobuf 时才导入，缺失时抛出带安装指引的清晰错误；
- 提供 ``describe_chakra_status()`` 供命令行在缺依赖时给出可读提示。

## 为什么需要 chakra

不是为了它的 ``chakra_converter``（那条链路依赖 PyTorch 的 host execution trace，
昇腾不产出；且它按 ``ncclKernel``/``record_param_comms`` 判类型，昇腾的
``hcom_*`` 一个都不匹配）。只需要它的两样东西：

1. ``et_def.proto`` 定义的 schema —— 这是 Chakra 生态的通用契约；
2. ``protolib.encode_message`` 的长度前缀格式 —— 与 ASTRA-sim 的 C++ ET feeder
   ``trace_.read(msg)`` 完全对齐。

节点构建完全由本项目负责，因此不继承官方转换器对 NCCL 名字的假设。
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from . import ir

#: Chakra schema 版本。与官方 ``chakra_converter`` 写出的 ``GlobalMetadata.version`` 一致。
CHAKRA_VERSION = "1.0.0"

#: 节点种类 -> Chakra NodeType 名称（写文件时按名字取枚举）。
KIND_TO_NODETYPE = {
    ir.COMP: "COMP_NODE",
    ir.COMM_COLL: "COMM_COLL_NODE",
    ir.COMM_SEND: "COMM_SEND_NODE",
    ir.COMM_RECV: "COMM_RECV_NODE",
    ir.METADATA: "METADATA_NODE",
}

#: 集合通信类型 -> Chakra CollectiveCommType 名称。
COMM_TYPE_TO_ENUM = {
    "all_reduce": "ALL_REDUCE",
    "reduce": "REDUCE",
    "all_gather": "ALL_GATHER",
    "gather": "GATHER",
    "scatter": "SCATTER",
    "broadcast": "BROADCAST",
    "all_to_all": "ALL_TO_ALL",
    "reduce_scatter": "REDUCE_SCATTER",
    "reduce_scatter_block": "REDUCE_SCATTER_BLOCK",
    "barrier": "BARRIER",
}

INSTALL_HINT = (
    "Chakra 未安装。它不在 PyPI 上，请用官方 git 源安装：\n"
    "    pip install git+https://github.com/mlcommons/chakra.git\n"
    "若目标节点离线，请在有网机器上构建 wheel 后分发，"
    "或把 et_def.proto / storage.proto / protolib.py 收进 3rdparty 自行生成 pb2。\n"
    "只输出 --format json 时不需要 Chakra。"
)


class ChakraUnavailable(RuntimeError):
    """chakra 或其 pb2 模块不可用。"""


def _import_et_def():
    """导入 ``chakra.schema.protobuf.et_def_pb2``。

    chakra 的 ``schema/protobuf/__init__.py`` 会把自身目录插进 ``sys.path``，因为
    protoc 生成的 ``et_def_pb2`` 用的是平铺的 ``import storage_pb2``。所以只要包
    正常安装，这里就能直接导入。
    """
    try:
        from chakra.schema.protobuf import et_def_pb2  # pylint: disable=import-outside-toplevel
    except Exception as error:  # pylint: disable=broad-except
        raise ChakraUnavailable(INSTALL_HINT) from error
    return et_def_pb2


def _encode_message(handle, message) -> None:
    """按 Chakra 官方格式写一条 message：varint 长度前缀 + 序列化字节。

    这里自己实现而不 import ``chakra.src.third_party.utils.protolib``，是因为
    ``chakra.src`` 不是公开入口，不同版本路径会变。格式极其简单（见官方
    protolib ``_EncodeVarint32``），自己实现反而更稳，同时保持一致。
    """
    payload = message.SerializeToString()
    value = len(payload)
    buffer = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            buffer.append(byte | 0x80)
        else:
            buffer.append(byte)
            break
    handle.write(bytes(buffer))
    handle.write(payload)


def describe_chakra_status() -> Dict[str, Any]:
    """探测 chakra 可用性，返回结构化状态（命令行和报告都用它）。"""
    status: Dict[str, Any] = {"available": False, "version": None, "path": None, "error": None}
    try:
        import chakra  # pylint: disable=import-outside-toplevel

        status["path"] = getattr(chakra, "__file__", None)
        status["version"] = getattr(chakra, "__version__", None)
    except Exception as error:  # pylint: disable=broad-except
        status["error"] = str(error)
        return status
    try:
        _import_et_def()
        status["available"] = True
    except ChakraUnavailable as error:
        status["error"] = str(error)
    return status


def _attr(et_def_pb2, name: str, **kwargs):
    """构造一个 AttributeProto。

    只允许一个字面量字段被设置（proto3 oneof 语义），因此调用方必须明确用
    ``int64_val`` / ``string_val`` / ``bool_val`` / ``double_val`` 之一。
    """
    attr = et_def_pb2.AttributeProto()
    attr.name = name
    for field_name, value in kwargs.items():
        setattr(attr, field_name, value)
    return attr


def _node_attrs(et_def_pb2, node: ir.Node) -> List[Any]:
    """把节点上需要审计的信息写进 ``attr``。

    Chakra 的 ``ETFeederNode`` 会把不认识的 attr 收进 ``other_attrs_``，可通过
    ``get_other_attr()`` 取回，因此追加自定义 attr 是官方支持的安全扩展，
    比塞进 ``inputs.values`` 之类的字符串字段更好。
    """
    attrs: List[Any] = []
    attrs.append(_attr(et_def_pb2, "rank", int64_val=int(node.rank)))
    attrs.append(_attr(et_def_pb2, "seq", int64_val=int(node.seq)))
    if isinstance(node.attrs.get("ranks"), list):
        attrs.append(_attr(et_def_pb2, "ranks", string_val=json.dumps(sorted(node.attrs["ranks"]))))
    timing = node.timing
    attrs.append(_attr(et_def_pb2, "time_source", string_val=str(timing.source)))
    if timing.comm_time_us is not None:
        attrs.append(_attr(et_def_pb2, "comm_time_us", double_val=float(timing.comm_time_us)))
    if timing.bandwidth_gbps is not None:
        attrs.append(_attr(et_def_pb2, "bandwidth_gbps", double_val=float(timing.bandwidth_gbps)))

    if node.comm is not None:
        comm = node.comm
        if comm.comm_type in COMM_TYPE_TO_ENUM:
            attrs.append(
                _attr(
                    et_def_pb2,
                    "comm_type",
                    int64_val=int(getattr(et_def_pb2, COMM_TYPE_TO_ENUM[comm.comm_type])),
                )
            )
        elif comm.comm_type:
            attrs.append(_attr(et_def_pb2, "comm_type_name", string_val=str(comm.comm_type)))
        if comm.nbytes is not None:
            attrs.append(_attr(et_def_pb2, "comm_size", int64_val=int(comm.nbytes)))
        if comm.group_name:
            attrs.append(_attr(et_def_pb2, "pg_name", string_val=str(comm.group_name)))
        if comm.comm_name:
            attrs.append(_attr(et_def_pb2, "comm_name", string_val=str(comm.comm_name)))
        if comm.participants:
            attrs.append(_attr(et_def_pb2, "participants", string_val=json.dumps(sorted(comm.participants))))
        if comm.dtype:
            attrs.append(_attr(et_def_pb2, "comm_dtype", string_val=str(comm.dtype)))
        if comm.count is not None:
            attrs.append(_attr(et_def_pb2, "comm_count", int64_val=int(comm.count)))
        if comm.src_rank is not None:
            attrs.append(_attr(et_def_pb2, "comm_src", int64_val=int(comm.src_rank)))
        if comm.dst_rank is not None:
            attrs.append(_attr(et_def_pb2, "comm_dst", int64_val=int(comm.dst_rank)))
        if comm.tag is not None:
            attrs.append(_attr(et_def_pb2, "comm_tag", int64_val=int(comm.tag)))

    for key in ("fused_group", "fused_order", "fused_role", "time_split"):
        value = node.attrs.get(key)
        if value:
            attrs.append(_attr(et_def_pb2, key, string_val=str(value)))
    if node.attrs.get("fused_ops"):
        attrs.append(_attr(et_def_pb2, "fused_ops", string_val=json.dumps(node.attrs["fused_ops"])))
    if node.attrs.get("fused_unexpanded"):
        attrs.append(_attr(et_def_pb2, "fused_unexpanded", bool_val=True))
    if node.attrs.get("dep_missing"):
        attrs.append(_attr(et_def_pb2, "dep_missing", string_val=json.dumps(node.attrs["dep_missing"])))
    return attrs


def build_chakra_messages(
    graph: ir.RankGraph,
    et_def_pb2,
    metadata_attrs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, List[Any]]:
    """把 ``RankGraph`` 翻译成 ``(GlobalMetadata, [Node, ...])``。

    关键映射：

    - ``start_time_micros`` / ``duration_micros`` 取整（schema 是 uint64），
      取整前已由 ``timing.normalize_timeline`` 平移到相对 0。
    - 依赖写在 ``data_deps``（仿真器读这个字段决定可执行顺序），
      ``ctrl_deps`` 留空——两者都填会让 feeder 的依赖计数出现"永不解除"。
    """
    metadata = et_def_pb2.GlobalMetadata()
    metadata.version = CHAKRA_VERSION
    attr_list: List[Any] = [_attr(et_def_pb2, "schema", string_val="Ascend")]
    for name, value in (metadata_attrs or {}).items():
        if value is None:
            continue
        if isinstance(value, bool):
            attr_list.append(_attr(et_def_pb2, name, bool_val=value))
        elif isinstance(value, (int, float)):
            attr_list.append(_attr(et_def_pb2, name, int64_val=int(value)))
        else:
            attr_list.append(_attr(et_def_pb2, name, string_val=str(value)))
    metadata.attr.extend(attr_list)

    # 先建立 id 映射：Chakra 要求 id 为 uint64 且唯一。字符串 id 用稳定哈希转数字，
    # 同时把原始 id 写进 attr 便于反查。
    id_map, reverse = _assign_numeric_ids(graph)

    messages: List[Any] = []
    for node in graph.nodes:
        message = et_def_pb2.Node()
        message.id = id_map[node.id]
        message.name = str(node.name)
        message.type = int(getattr(et_def_pb2, KIND_TO_NODETYPE[node.kind]))
        if node.timing.start_us is not None:
            message.start_time_micros = int(round(max(node.timing.start_us, 0.0)))
        if node.timing.duration_us is not None:
            message.duration_micros = int(round(max(node.timing.duration_us, 0.0)))
        for pred_id in node.preds:
            numeric = id_map.get(pred_id)
            if numeric is not None:
                message.data_deps.append(numeric)
        message.attr.append(_attr(et_def_pb2, "chakra_node_id", string_val=str(node.id)))
        message.attr.extend(_node_attrs(et_def_pb2, node))
        messages.append(message)

    graph.meta.setdefault("id_mapping_count", len(reverse))
    return metadata, messages


def _assign_numeric_ids(graph: ir.RankGraph) -> Tuple[Dict[str, int], Dict[int, str]]:
    """字符串节点 id -> 唯一 uint64。

    不直接用 ``hash()``（受 PYTHONHASHSEED 影响），而是用确定性的递增分配：
    排序后依次编号。这样同一份输入两次运行的输出完全一致，便于 diff 和回归测试。
    """

    def sort_key(node: ir.Node):
        return (node.rank if node.rank >= 0 else 10**9, node.seq, node.id)

    ordered = sorted(graph.nodes, key=sort_key)
    id_map: Dict[str, int] = {}
    reverse: Dict[int, str] = {}
    for index, node in enumerate(ordered, start=1):
        id_map[node.id] = index
        reverse[index] = node.id
    return id_map, reverse


def write_chakra_protobuf(
    graph: ir.RankGraph,
    output_path: str,
    metadata_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """写一个 ``.et`` 文件，返回统计信息。

    文件布局与官方一致：``GlobalMetadata`` 在前，随后逐条 ``Node``，
    每条都带 varint 长度前缀。
    """
    et_def_pb2 = _import_et_def()
    metadata, messages = build_chakra_messages(graph, et_def_pb2, metadata_attrs)

    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "wb") as handle:
        _encode_message(handle, metadata)
        for message in messages:
            _encode_message(handle, message)
    return {
        "path": output_path,
        "nodes": len(messages),
        "version": CHAKRA_VERSION,
        "bytes": os.path.getsize(output_path),
    }


def read_chakra_file(path: str, limit: Optional[int] = None) -> Tuple[Any, List[Any]]:
    """读回一个 ``.et`` 文件用于自检。

    实现 varint 解码，不依赖 chakra 的 ``protolib``，因此即使只装了 pb2 也能校验。
    """
    et_def_pb2 = _import_et_def()
    messages: List[Any] = []
    with open(path, "rb") as handle:
        payload = handle.read()
    offset = 0
    metadata = None
    while offset < len(payload):
        size, offset = _decode_varint(payload, offset)
        if size <= 0:
            break
        chunk = payload[offset : offset + size]
        offset += size
        if metadata is None:
            metadata = et_def_pb2.GlobalMetadata()
            metadata.ParseFromString(chunk)
            continue
        node = et_def_pb2.Node()
        node.ParseFromString(chunk)
        messages.append(node)
        if limit is not None and len(messages) >= limit:
            break
    return metadata, messages


def _decode_varint(payload: bytes, offset: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while offset < len(payload):
        byte = payload[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, offset
        shift += 7
        if shift >= 64:
            raise ValueError("varint 过长，文件可能不是 Chakra 格式")
    return 0, offset


def graph_to_json(graph: ir.RankGraph, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把图转成可直接 json.dump 的字典。

    这是合并图的交付形态：人类可读、可 diff，且保留每条边的来源
    （``pred_sources``），能回答"这条依赖凭什么存在"。protobuf 用于下游仿真器，
    JSON 用于审计。
    """
    payload = {
        "schema": "mindspeed-llm/dag/v1",
        "chakra_version": CHAKRA_VERSION,
        "rank": graph.rank,
        "base_ts_us": graph.base_ts_us,
        "meta": dict(graph.meta),
        "nodes": [],
    }
    for node in graph.nodes:
        payload["nodes"].append(
            {
                "id": node.id,
                "kind": node.kind,
                "name": node.name,
                "rank": node.rank,
                "seq": node.seq,
                "preds": list(node.preds),
                "pred_sources": dict(node.pred_sources),
                "succs": list(node.succs),
                "comm": node.comm.to_dict() if node.comm else None,
                "timing": node.timing.to_dict(),
                "attrs": dict(node.attrs),
            }
        )
    if extra:
        payload["extra"] = extra
    return payload


def write_json(payload: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    return {"path": output_path, "bytes": os.path.getsize(output_path)}
