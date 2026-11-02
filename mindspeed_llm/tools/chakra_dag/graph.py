# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""由运行时事件构建 DAG，并完成多卡 collective 合并与图校验。

本模块是纯算法，不依赖 torch / chakra，因此可以在个人电脑上完整单元测试。

核心不变量：

1. **依赖只来自事件记录**，绝不按时间戳排序造边。``Event.deps`` 是采集期的观测结果。
2. **边去重**：``Node.add_pred`` 保证同一 (src, dst) 只出现一次。
3. **无环**：合并完成后做拓扑排序校验，有环直接报错而不是输出坏图。
4. **跨卡 collective 合并是确定性匹配**：主键是
   ``(group_name, comm_type, size, dtype, occurrence)``，其中 ``occurrence`` 是
   采集期分配的序号。SPMD 下各 rank 的调用序列一致，所以这等价于按序号对齐，
   不需要"逻辑组 + 类型 + count + 时间窗重叠"这类启发式。
"""

import hashlib
import json
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import ir


class GraphError(RuntimeError):
    """图不合法（环、悬空边、字段冲突等）。"""


def _node_id(rank: int, local: int) -> str:
    return f"r{rank}:{local}"


def _collective_id(key: Tuple) -> str:
    """由匹配键生成稳定的合并节点 id。

    用短哈希而不是原始键，是为了让 id 保持短小且不含空格/冒号，
    同时保留键本身写进 ``attrs["match_key"]`` 便于排查。
    """
    payload = json.dumps(list(key), ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"coll:{digest}"


def build_rank_graph(
    rank: int,
    events: Sequence[ir.Event],
    base_ts_us: Optional[float] = None,
    meta: Optional[Dict] = None,
) -> ir.RankGraph:
    """把一个 rank 的事件序列转成节点图。

    输入：
        rank：全局 rank。
        events：按采集顺序排列的事件；``Event.seq`` 决定顺序。
        base_ts_us：该 rank 采集窗口的绝对起点，仅记录，不参与建边。
        meta：rank 级附加信息（device_id、并行组、trace 路径等）。

    输出：
        ``RankGraph``，节点 id 形如 ``r<rank>:<local>``。

    处理内容：
        1. 按 ``seq`` 排序事件（不按时间戳）。
        2. 逐个事件翻译成节点；融合算子展开成多个兄弟节点。
        3. 把 ``Event.deps`` 解析成前驱边：若被依赖事件展开成多个节点，
           本节点依赖其**全部**产出节点，等价于等待整个融合体完成。
        4. 反推 succs，并在报告里记录展开/丢弃情况。
    """
    ordered = sorted(events, key=lambda item: item.seq)

    visited_seq: Set[int] = set()
    for event in ordered:
        if event.seq in visited_seq:
            raise GraphError(f"duplicate event seq {event.seq} on rank {rank}")
        visited_seq.add(event.seq)

    produced: Dict[str, List[str]] = {}
    nodes: List[ir.Node] = []
    by_id: Dict[str, ir.Node] = {}
    local = 0
    expanded = 0
    unresolved_fused: List[str] = []

    for event in ordered:
        new_nodes = _translate_event(event, rank, local, produced, unresolved_fused)
        for node in new_nodes:
            by_id[node.id] = node
            nodes.append(node)
        local += len(new_nodes)
        produced[event.eid] = [node.id for node in new_nodes]
        if event.kind == ir.EVENT_FUSED and len(new_nodes) > 1:
            expanded += 1

    graph = ir.RankGraph(rank=rank, nodes=nodes, base_ts_us=base_ts_us, meta=dict(meta or {}))
    graph.index_succs()
    graph.meta.setdefault("stats", {})
    graph.meta["stats"].update(
        {
            "nodes": len(nodes),
            "events": len(ordered),
            "fused_expanded": expanded,
            "unresolved_fused": sorted(set(unresolved_fused)),
        }
    )
    return graph


def _translate_event(
    event: ir.Event,
    rank: int,
    local: int,
    produced: Dict[str, List[str]],
    unresolved_fused: List[str],
) -> List[ir.Node]:
    """单个事件 -> 一个或多个节点。"""
    preds: List[str] = []
    pred_sources: Dict[str, str] = {}
    missing_deps: List[str] = []
    for dep_eid, dep_source in zip(event.deps, event.dep_sources):
        dep_nodes = produced.get(dep_eid)
        if not dep_nodes:
            missing_deps.append(dep_eid)
            continue
        for dep_id in dep_nodes:
            if dep_id in pred_sources:
                continue
            preds.append(dep_id)
            pred_sources[dep_id] = dep_source

    def make(kind: str, name: str, index: int, **kwargs) -> ir.Node:
        node = ir.Node(
            id=_node_id(rank, local + index),
            kind=kind,
            name=name,
            rank=rank,
            seq=local + index,
            preds=list(preds),
            pred_sources=dict(pred_sources),
            attrs=dict(kwargs.get("attrs", {})),
        )
        node.timing = kwargs.get("timing", ir.Timing())
        node.comm = kwargs.get("comm")
        return node

    if event.kind == ir.EVENT_COMPUTE:
        attrs = dict(event.attrs)
        if missing_deps:
            attrs["missing_deps"] = missing_deps
        return [make(ir.COMP, event.name, 0, attrs=attrs)]

    if event.kind == ir.EVENT_COMM_COLL:
        attrs = dict(event.attrs)
        if missing_deps:
            attrs["missing_deps"] = missing_deps
        return [make(ir.COMM_COLL, event.name, 0, comm=event.comm, attrs=attrs)]

    if event.kind == ir.EVENT_P2P_SEND:
        return [make(ir.COMM_SEND, event.name, 0, comm=event.comm, attrs=dict(event.attrs))]

    if event.kind == ir.EVENT_P2P_RECV:
        return [make(ir.COMM_RECV, event.name, 0, comm=event.comm, attrs=dict(event.attrs))]

    if event.kind != ir.EVENT_FUSED:
        raise GraphError(f"unsupported event kind: {event.kind}")

    spec = event.fused
    if spec is None:
        # 名字看起来是融合算子，但规则表里没有。退化为单个计算节点并记录下来，
        # 绝不静默当作普通算子——报告里会让调用方看到。
        unresolved_fused.append(event.name)
        attrs = dict(event.attrs)
        attrs.update({"fused_unexpanded": True, "fused_ops": []})
        return [make(ir.COMP, event.name, 0, attrs=attrs)]

    fused_id = f"fused:{event.eid}"
    nodes: List[ir.Node] = []
    index = 0
    for op_name in spec.ops:
        is_comm = op_name in ir.COMM_TYPES
        attrs = dict(event.attrs)
        attrs.update(
            {
                "fused_group": fused_id,
                "fused_ops": list(spec.ops),
                "fused_order": spec.order,
                "fused_role": "comm" if is_comm else "comp",
                # 融合体内的兄弟节点共享前驱/后继，不能串成链。
                "fused_siblings": [],
            }
        )
        if is_comm:
            comm = ir.CommMeta(
                comm_type=spec.comm_type or op_name,
                group_name=spec.group_name,
                comm_name=spec.comm_name,
                count=spec.count,
                dtype=spec.dtype,
                nbytes=spec.nbytes,
                participants=spec.participants,
                occurrence=spec.occurrence,
                fused_with=fused_id,
                fused_role="comm",
            )
            node = make(ir.COMM_COLL, op_name, index, comm=comm, attrs=attrs)
        else:
            node = make(ir.COMP, op_name, index, attrs=attrs)
        nodes.append(node)
        index += 1

    sibling_ids = [node.id for node in nodes]
    for node in nodes:
        node.attrs["fused_siblings"] = [item for item in sibling_ids if item != node.id]
    return nodes


def find_pending_collective_merges(
    graphs: Sequence[ir.RankGraph],
) -> Tuple[Dict[Tuple, List[Tuple[int, ir.Node]]], List[Tuple[int, ir.Node]]]:
    """扫描各 rank 的图，找出可以跨卡合并的 collective 节点。

    输出：
        ``(mergeable, unmergeable)``。``mergeable`` 是 匹配键 -> [(rank, node)]；
        ``unmergeable`` 是无法安全配对的 collective（缺少 group/type/size），
        这些节点会保留为 rank-local 并在报告中列出。
    """
    mergeable: "OrderedDict[Tuple, List[Tuple[int, ir.Node]]]" = OrderedDict()
    unmergeable: List[Tuple[int, ir.Node]] = []
    for graph in graphs:
        for node in graph.nodes:
            if node.kind != ir.COMM_COLL:
                continue
            key = node.comm.match_key() if node.comm else None
            if key is None:
                unmergeable.append((graph.rank, node))
                continue
            mergeable.setdefault(key, []).append((graph.rank, node))
    return mergeable, unmergeable


def merge_multi_rank(
    graphs: Sequence[ir.RankGraph],
    strict: bool = True,
) -> Tuple[ir.RankGraph, Dict]:
    """把多个 rank 的图合并成一张图，跨卡 collective 合并为单节点。

    输入：
        graphs：各 rank 的图。
        strict：True 时有环/悬空边直接抛错；False 时只记录在报告里。

    输出：
        ``(merged_graph, report)``。合并图用 rank ``-1`` 表示"整机"，
        每个节点都用 ``attrs["ranks"]`` 记录它实际来自哪些卡。

    处理内容：
        1. 按匹配键把同一 collective 在各 rank 上的节点合并，id 统一为 ``coll:<hash>``。
        2. 把原图中指向被合并节点的边改指到合并节点，并去重。
        3. 合并节点的时间取包络：``start=min``，``end=max(ts+dur)``。
        4. 每个参与 rank 的原始记录保存在 ``attrs["sources"]`` 里以便审计。
        5. 拓扑排序检查环。
    """
    mergeable, unmergeable = find_pending_collective_merges(graphs)

    # 1) 建立旧 id -> 新 id 的映射，以及合并节点的内容。
    id_map: Dict[str, str] = {}
    merged_nodes: Dict[str, ir.Node] = {}
    merged_sources: Dict[str, List[Dict]] = {}

    for key, members in mergeable.items():
        if len({rank for rank, _ in members}) < 2:
            # 只有一个 rank 参与，说明不是真正的跨卡 collective（或数据不全），
            # 保持原样，避免制造虚假的"跨卡"节点。
            continue
        new_id = _collective_id(key)
        ranks = sorted({rank for rank, _ in members})
        first = members[0][1]
        collectives = [node.comm for _, node in members if node.comm is not None]
        participants: List[int] = []
        for comm in collectives:
            for item in comm.participants or []:
                if item not in participants:
                    participants.append(item)
        participants = sorted(participants) if participants else ranks

        starts = [node.timing.start_us for _, node in members if node.timing.start_us is not None]
        ends = [
            node.timing.start_us + node.timing.duration_us
            for _, node in members
            if node.timing.start_us is not None and node.timing.duration_us is not None
        ]
        comm_times = [node.timing.comm_time_us for _, node in members if node.timing.comm_time_us is not None]
        bandwidths = [node.timing.bandwidth_gbps for _, node in members if node.timing.bandwidth_gbps is not None]

        merged_comm = ir.CommMeta(
            comm_type=first.comm.comm_type if first.comm else "",
            group_name=first.comm.group_name if first.comm else "",
            comm_name=first.comm.comm_name if first.comm else "",
            count=first.comm.count if first.comm else None,
            dtype=first.comm.dtype if first.comm else None,
            nbytes=first.comm.nbytes if first.comm else None,
            participants=participants,
            occurrence=first.comm.occurrence if first.comm else 0,
            fused_with=first.comm.fused_with if first.comm else None,
            fused_role=first.comm.fused_role if first.comm else None,
        )
        node = ir.Node(
            id=new_id,
            kind=ir.COMM_COLL,
            name=first.name,
            rank=-1,
            seq=min(item.seq for _, item in members),
            comm=merged_comm,
            attrs={
                "match_key": list(key),
                "ranks": ranks,
                "sources": [],
                "fused_group": first.attrs.get("fused_group"),
            },
        )
        # 融合相关的属性必须继承下来，否则合并后就看不出"这段通信属于哪个融合体、
        # 在融合体里排第几、时间是测的还是估的"。
        for inherited in ("fused_ops", "fused_order", "fused_role", "time_split", "fused_siblings"):
            if inherited in first.attrs:
                node.attrs[inherited] = first.attrs[inherited]
        if first.attrs.get("fused_unexpanded"):
            node.attrs["fused_unexpanded"] = True
        if starts:
            node.timing.start_us = min(starts)
        if starts and ends:
            node.timing.duration_us = max(ends) - min(starts)
            # 时间来源取各参与 rank 的"最弱"结论：只要有一个是估算，就标 estimated，
            # 避免把估算值伪装成实测值。
            sources = {member.timing.source for _, member in members}
            node.timing.source = ir.TIME_MEASURED if sources == {ir.TIME_MEASURED} else ir.TIME_ESTIMATED
        if comm_times:
            node.timing.comm_time_us = max(comm_times)
        if bandwidths:
            node.timing.bandwidth_gbps = max(bandwidths)

        merged_nodes[new_id] = node
        for rank, member in members:
            id_map[member.id] = new_id
            merged_sources.setdefault(new_id, []).append(
                {
                    "rank": rank,
                    "node_id": member.id,
                    "name": member.name,
                    "start_us": member.timing.start_us,
                    "duration_us": member.timing.duration_us,
                }
            )

    # 2) 复制所有未被合并的节点，同时把前驱边改指到合并后的 id。
    #    original_by_id 保留原始节点，用于给合并节点收集前驱。
    original_by_id: Dict[str, ir.Node] = {}
    for graph in graphs:
        for node in graph.nodes:
            original_by_id[node.id] = node

    out_nodes: List[ir.Node] = []
    for graph in graphs:
        for node in graph.nodes:
            if node.id in id_map:
                continue
            clone = ir.Node.from_dict(node.to_dict())
            clone.attrs.setdefault("ranks", [graph.rank])
            clone.preds = []
            clone.pred_sources = {}
            for pred_id, source in node.pred_sources.items():
                mapped = id_map.get(pred_id, pred_id)
                if mapped == clone.id:
                    continue
                clone.add_pred(mapped, source)
            out_nodes.append(clone)

    # 3) 合并节点的前驱 = 各参与 rank 上该 collective 前驱的并集（去重）。
    for new_id, members in merged_sources.items():
        node = merged_nodes[new_id]
        node.attrs["sources"] = list(members)
        for record in members:
            member = original_by_id.get(record["node_id"])
            if member is None:
                continue
            for pred_id, source in member.pred_sources.items():
                mapped = id_map.get(pred_id, pred_id)
                if mapped == new_id:
                    continue  # 同一 collective 的分片互相依赖，去掉自环
                node.add_pred(mapped, source)

    all_nodes = list(out_nodes) + list(merged_nodes.values())
    all_nodes.sort(key=lambda item: (item.rank if item.rank >= 0 else 10**6, item.seq))

    # 4) 悬空边检查。
    known = {node.id for node in all_nodes}
    dangling: List[Tuple[str, str]] = []
    for node in all_nodes:
        kept: List[str] = []
        kept_src: Dict[str, str] = {}
        for pred_id in node.preds:
            if pred_id in known:
                kept.append(pred_id)
                kept_src[pred_id] = node.pred_sources.get(pred_id, ir.DEP_TENSOR)
            else:
                dangling.append((node.id, pred_id))
        node.preds = kept
        node.pred_sources = kept_src

    merged = ir.RankGraph(rank=-1, nodes=all_nodes, base_ts_us=None, meta={})
    merged.index_succs()
    # 合并后可能出现冗余的传递前驱边（既有 B 又有 B 的上游 C）。去掉它们，
    # 否则仿真器会把启动条件判得过严。菱形依赖（兄弟节点共同前驱）不受影响。
    reduced = reduce_transitive_edges(merged)
    cycles = find_cycles(merged)
    if cycles and strict:
        raise GraphError(f"merged graph contains {len(cycles)} cycle(s); first={cycles[0]}")

    report = {
        "ranks": sorted({graph.rank for graph in graphs}),
        "nodes_total": len(all_nodes),
        "collectives_merged": len(merged_nodes),
        "collectives_unmergeable": len(unmergeable),
        "unmergeable_detail": [
            {"rank": rank, "node_id": node.id, "name": node.name} for rank, node in unmergeable[:50]
        ],
        "dangling_edges": len(dangling),
        "dangling_detail": dangling[:50],
        "transitive_edges_removed": reduced,
        "cycles": len(cycles),
        "cycle_detail": cycles[:5],
        "kind_counts": count_kinds(all_nodes),
    }
    return merged, report


def reduce_transitive_edges(graph: ir.RankGraph) -> int:
    """删除冗余的传递前驱边，返回删除数量。

    合并跨卡 collective 之后，会出现这种情况：节点 A 的直接前驱里既有 B，
    又有 B 的上游 C。此时 C 边是冗余的——等 B 完成时 C 必然已完成。

    冗余边本身不会造成错图，但会让仿真器把启动条件判得过严（要等多余的依赖），
    从而高估停顿。这里做一次传递归约：

    对每个节点，若它的两个前驱 u、v 之间存在 u 可达 v 的路径，则 u 是冗余的。

    实现上对每个节点做一次有限深度 DFS：节点数在万级以内，且每个节点的前驱
    通常很少，代价可接受。
    """
    by_id = graph.node_map()
    removed = 0
    for node in graph.nodes:
        if len(node.preds) < 2:
            continue
        ancestors: Dict[str, Set[str]] = {}
        for pred_id in node.preds:
            ancestors[pred_id] = _ancestors(by_id, pred_id, limit=4096)
        keep: List[str] = []
        for pred_id in node.preds:
            redundant = False
            for other in node.preds:
                if other == pred_id:
                    continue
                if pred_id in ancestors.get(other, ()):
                    redundant = True
                    break
            if redundant:
                removed += 1
                continue
            keep.append(pred_id)
        if len(keep) != len(node.preds):
            node.preds = keep
            node.pred_sources = {key: value for key, value in node.pred_sources.items() if key in keep}
    if removed:
        graph.index_succs()
    return removed


def _ancestors(by_id: Dict[str, ir.Node], start: str, limit: int = 4096) -> Set[str]:
    """收集 start 的所有祖先（含自身）。超过 limit 个时提前返回，避免病态图拖慢。"""
    result: Set[str] = set()
    stack = [start]
    while stack:
        node_id = stack.pop()
        if node_id in result:
            continue
        result.add(node_id)
        node = by_id.get(node_id)
        if node is None:
            continue
        stack.extend(node.preds)
        if len(result) > limit:
            break
    return result


def count_kinds(nodes: Iterable[ir.Node]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in nodes:
        counts[node.kind] = counts.get(node.kind, 0) + 1
    return counts


def find_cycles(graph: ir.RankGraph) -> List[List[str]]:
    """用迭代式 DFS 找环，返回若干条环路径（用于报错）。"""
    by_id = graph.node_map()
    white, gray, black = 0, 1, 2
    color = {node.id: white for node in graph.nodes}
    cycles: List[List[str]] = []

    for start in by_id:
        if color[start] != white:
            continue
        stack: List[Tuple[str, int]] = [(start, 0)]
        path: List[str] = []
        on_stack: List[str] = []
        while stack:
            node_id, index = stack[-1]
            if index == 0:
                color[node_id] = gray
                path.append(node_id)
                on_stack.append(node_id)
            node = by_id.get(node_id)
            succs = node.succs if node is not None else []
            if index < len(succs):
                stack[-1] = (node_id, index + 1)
                child = succs[index]
                if color.get(child) == gray:
                    if child in on_stack:
                        begin = on_stack.index(child)
                        cycles.append(on_stack[begin:] + [child])
                        if len(cycles) >= 20:
                            return cycles
                elif color.get(child) == white:
                    stack.append((child, 0))
            else:
                color[node_id] = black
                stack.pop()
                if path:
                    path.pop()
                if on_stack:
                    on_stack.pop()
    return cycles


def topological_order(graph: ir.RankGraph) -> List[str]:
    """Kahn 拓扑排序；有环时返回已排出的部分。"""
    indeg = {node.id: len(node.preds) for node in graph.nodes}
    ready = [node.id for node in graph.nodes if indeg[node.id] == 0]
    by_id = graph.node_map()
    order: List[str] = []
    while ready:
        node_id = ready.pop()
        order.append(node_id)
        for succ in by_id[node_id].succs:
            indeg[succ] -= 1
            if indeg[succ] == 0:
                ready.append(succ)
    return order


def validate(graph: ir.RankGraph, require_participants: bool = False) -> Dict:
    """校验图的完整性，返回检查结果字典（不抛错，由调用方决定严重性）。"""
    problems: List[str] = []
    seen: Set[str] = set()
    for node in graph.nodes:
        if node.id in seen:
            problems.append(f"duplicate node id {node.id}")
        seen.add(node.id)
        for pred_id in node.preds:
            if pred_id not in seen and pred_id not in {item.id for item in graph.nodes}:
                problems.append(f"{node.id} has unknown pred {pred_id}")
        if node.kind == ir.COMM_COLL:
            if node.comm is None:
                problems.append(f"{node.id} is collective but has no comm meta")
            elif require_participants and not node.comm.participants:
                problems.append(f"{node.id} collective has empty participants")
    cycles = find_cycles(graph)
    return {
        "problems": problems,
        "problem_count": len(problems),
        "cycles": len(cycles),
        "acyclic": not cycles,
        "duplicate_ids": len(graph.nodes) - len(seen),
    }
