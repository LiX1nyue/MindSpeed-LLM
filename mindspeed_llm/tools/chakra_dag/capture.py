# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""训练期运行时采集：把"结构"直接观测下来。

## 为什么不在事后从 trace 反推

``trace_view.json`` 只记录"谁在什么时候跑了多久"，不记录"谁依赖谁"。事后反推会
遇到一批无法确定的问题：跨卡集合通信怎么配对、跨流何时可以连边、计算与通信到底
谁等谁。这些信息在训练时本来就在进程内，只是没有被记录。

本模块做三件事：

1. **计算算子**：用 ``TorchDispatchMode`` 拦截每一次 Aten 调用，记录被读写的张量
   ``(id(tensor), _version)``。两个算子共用同一份未被改写的张量时，就存在一条真实
   的数据依赖——这是观测，不是推断。
2. **非融合通信**：包装 ``torch.distributed`` 上的集合通信与 P2P 入口，记录组、
   类型、dtype、count 以及**在算子图中的位置**。异步调用的完成点（``wait()``）
   也会被记录，因为真实同步发生在那里，而不是入队处。
3. **融合算子**：拦截 ``torch_npu`` 与 MindSpeed ``coc_ops`` 上的融合入口，交给
   ``fused_ops`` 规则表展开成「通信 + 计算」两个兄弟节点。

## 已知边界（必须如实告知使用者）

- 只看得到 Aten 层算子。被融合进单个 kernel 的计算不会单独出现。
- 融合算子的**内部时长切分**拿不到，只能靠 trace 实测或带宽估算，见 ``timing``。
- 打开后有明显性能损耗，只适合采集 1~2 个 step。

## 不依赖 torch 就能 import

模块顶层不 import torch，所有 torch 相关符号都在函数内导入。这样离线合并、
单元测试在本机没有 torch 时也能导入本包。
"""

import contextlib
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import fused_ops, ir

# ---------------------------------------------------------------------------
# 分布式入口清单：collective 名字 -> (方法名, 是否是集合通信)
# ---------------------------------------------------------------------------
# 说明：只包装 torch.distributed 上存在的名字，缺失的会被跳过，
# 这样不同 torch 版本可以共用同一张表。
_COLLECTIVE_APIS: Dict[str, str] = {
    "all_reduce": "all_reduce",
    "all_reduce_coalesced": "all_reduce",
    "reduce": "reduce",
    "all_gather": "all_gather",
    "all_gather_into_tensor": "all_gather",
    "all_gather_coalesced": "all_gather",
    "_all_gather_base": "all_gather",
    "reduce_scatter": "reduce_scatter",
    "reduce_scatter_tensor": "reduce_scatter",
    "_reduce_scatter_base": "reduce_scatter",
    "all_to_all": "all_to_all",
    "all_to_all_single": "all_to_all",
    "broadcast": "broadcast",
    "broadcast_object_list": "broadcast",
    "scatter": "scatter",
    "gather": "gather",
    "barrier": "barrier",
}

_P2P_APIS: Dict[str, str] = {
    "send": "p2p_send",
    "recv": "p2p_recv",
    "isend": "p2p_send",
    "irecv": "p2p_recv",
}

#: functional collectives 在 Aten 层可见，但它们的语义也应作为通信节点进图。
#: 名字 -> collective 类型。用于 dispatch 回调里的识别。
_FUNCTIONAL_COLLECTIVE_PREFIX = "_c10d_functional::"


def _torch():
    """延迟导入 torch，缺失时给出可读的错误。"""
    try:
        import torch  # noqa: F401  pylint: disable=import-outside-toplevel
    except ImportError as error:  # pragma: no cover - 本机无 torch
        raise RuntimeError("chakra_dag.capture 需要 torch。离线合并/写出不需要 torch，只有训练期采集需要。") from error
    return torch


# ---------------------------------------------------------------------------
# 张量哈希：用 (id, version) 而不是内容，成本 O(1)
# ---------------------------------------------------------------------------
def _tensor_keys(obj: Any, torch_mod) -> List[Tuple[int, int]]:
    """递归提取对象里所有 torch 张量的 ``(id, _version)``。

    ``_version`` 在就地写时自增，因此 ``(id, version)`` 能区分"同一块的旧内容"
    和"被改写后的新内容"，这正是数据依赖需要的语义。
    """
    found: List[Tuple[int, int]] = []

    def walk(item, depth):
        if depth > 4:
            return
        if isinstance(item, torch_mod.Tensor):
            try:
                version = item._version
            except Exception:  # pylint: disable=broad-except
                version = -1
            found.append((id(item), int(version)))
        elif isinstance(item, (list, tuple)):
            for sub in item:
                walk(sub, depth + 1)
        elif isinstance(item, dict):
            for sub in item.values():
                walk(sub, depth + 1)

    walk(obj, 0)
    return found


def _describe_tensor(tensor, name: str) -> Optional[fused_ops.ArgValue]:
    """把一个 torch 张量翻译成 ``ArgValue``。"""
    if tensor is None or not hasattr(tensor, "numel"):
        return None
    try:
        numel = int(tensor.numel())
        dtype = str(tensor.dtype).replace("torch.", "")
        shape = [int(item) for item in tensor.shape]
    except Exception:  # pylint: disable=broad-except
        return None
    return fused_ops.ArgValue(name=name, numel=numel, dtype=dtype, shape=shape)


def _arg_value(value: Any, name: str) -> Optional[fused_ops.ArgValue]:
    """把任意实参翻译成 ``ArgValue``：张量取 shape/dtype，字符串取 text。"""
    if isinstance(value, str):
        return fused_ops.ArgValue(name=name, text=value)
    if value is None:
        return None
    if hasattr(value, "numel") and hasattr(value, "dtype"):
        return _describe_tensor(value, name)
    if isinstance(value, (list, tuple)) and value and hasattr(value[0], "numel"):
        # 张量列表（all_gather 之类）：累加 numel。
        total = 0
        dtype = None
        for item in value:
            described = _describe_tensor(item, name)
            if described is None:
                continue
            total += described.numel or 0
            dtype = dtype or described.dtype
        if total:
            return fused_ops.ArgValue(name=name, numel=total, dtype=dtype)
    return None


# ---------------------------------------------------------------------------
# 采集器主体
# ---------------------------------------------------------------------------
class DagCapture:
    """运行时采集器。

    用法（训练脚本内）::

        cap = DagCapture(rank=dist.get_rank(), world_size=dist.get_world_size())
        cap.install()
        ...  # 跑 1~2 个 step
        cap.uninstall()
        cap.dump(profile_dir)

    设计取舍：
        - **只保留区间内的事件**：``window_start`` / ``window_end`` 用单调时钟，
          避免采集到初始化阶段。
        - **先记录再裁剪**：dispatch 回调里只做最小工作，裁剪留到 dump 时。
        - **失败不影响训练**：``install`` 中任何一步失败都静默降级并置
          ``self.degraded``，由导出层决定是否告警。
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        max_events: int = 2_000_000,
        capture_tensor_keys: bool = True,
    ):
        self.rank = rank
        self.world_size = world_size
        self.max_events = max_events
        self.capture_tensor_keys = capture_tensor_keys

        self.events: List[ir.Event] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._installed = False
        self._degraded = False
        self._notes: List[str] = []

        # 张量 key -> 最近一次写入它的 event eid
        self._last_writer: Dict[Tuple[int, int], str] = {}
        # 当前 Python 线程上的"上一个事件"，提供同线程顺序依赖
        self._thread_last: Dict[int, str] = {}
        # 未完成写的张量 key 集合，用于识别就地写
        self._written_keys: Dict[str, List[Tuple[int, int]]] = {}

        # 每个逻辑组上各 collective 类型的出现计数，跨卡配对主键的一半
        self._occurrence: Dict[Tuple[str, str], int] = {}
        # 待完成的异步通信：handle id -> eid
        self._pending_comm: Dict[int, str] = {}

        self._originals: List[Tuple[Any, str, Any]] = []
        self._dispatch_handle = None
        self._torch = None
        self._group_rank_cache: Dict[int, List[int]] = {}
        self._base_ts_us: Optional[float] = None
        self._window_start: Optional[float] = None
        self._window_end: Optional[float] = None

    # -- 安装 / 卸载 ------------------------------------------------------
    def install(self) -> bool:
        """安装所有钩子。返回是否全部成功（部分成功时 ``degraded`` 为 True）。"""
        if self._installed:
            return True
        try:
            self._torch = _torch()
        except RuntimeError as error:
            self._notes.append(str(error))
            self._degraded = True
            return False

        ok = True
        ok &= self._install_distributed_hooks()
        ok &= self._install_fused_hooks()
        ok &= self._install_dispatch_mode()
        self._installed = True
        self._base_ts_us = time.time() * 1e6
        self._window_start = time.monotonic()
        return ok

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._window_end = time.monotonic()
        # TorchDispatchMode 是栈式的：必须显式 __exit__，否则即使属性被还原，
        # 采集回调仍会在每次算子调用时触发，训练会一直背着这个开销。
        if self._dispatch_handle is not None:
            try:
                self._dispatch_handle.__exit__(None, None, None)
            except Exception:  # pylint: disable=broad-except
                pass
            self._dispatch_handle = None
        for target, name, original in reversed(self._originals):
            try:
                setattr(target, name, original)
            except Exception:  # pylint: disable=broad-except
                pass
        self._originals.clear()
        self._installed = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def notes(self) -> List[str]:
        return list(self._notes)

    # -- 分布式钩子 ------------------------------------------------------
    def _install_distributed_hooks(self) -> bool:
        torch = self._torch
        dist = getattr(torch, "distributed", None)
        if dist is None or not getattr(dist, "is_available", lambda: False)():
            self._notes.append("torch.distributed 不可用，跳过通信采集")
            self._degraded = True
            return False

        installed = 0
        for name, comm_type in _COLLECTIVE_APIS.items():
            original = getattr(dist, name, None)
            if original is None:
                continue
            self._originals.append((dist, name, original))
            setattr(dist, name, self._make_collective_wrapper(original, name, comm_type))
            installed += 1

        for name, kind in _P2P_APIS.items():
            original = getattr(dist, name, None)
            if original is None:
                continue
            self._originals.append((dist, name, original))
            setattr(dist, name, self._make_p2p_wrapper(original, name, kind))
            installed += 1

        # batch_isend_irecv 是 PP 的主路径，单独包一层以便识别成对 P2P。
        original = getattr(dist, "batch_isend_irecv", None)
        if original is not None:
            self._originals.append((dist, "batch_isend_irecv", original))
            setattr(dist, "batch_isend_irecv", self._make_batch_p2p_wrapper(original))
            installed += 1

        if installed == 0:
            self._notes.append("没有找到任何 torch.distributed 通信入口")
            self._degraded = True
        return installed > 0

    def _resolve_group(self, group) -> Tuple[str, Optional[List[int]]]:
        """把 process group 对象翻译成 ``(组名, global rank 列表)``。

        组名优先用 HCCL 通信域名字——它通常出现在 profiling 的
        ``hcom_<type>__<name>_<seq>`` 里，是把运行时记录和 trace 实测时长对上号的
        关键。取不到时退化为组内 rank 列表拼成的字符串。
        """
        torch = self._torch
        dist = torch.distributed
        if group is None:
            try:
                group = dist.group.WORLD
            except Exception:  # pylint: disable=broad-except
                return "", None
        cached = self._group_rank_cache.get(id(group))
        if cached is not None:
            return self._group_rank_cache[id(group)]  # type: ignore[return-value]

        ranks: Optional[List[int]] = None
        name = ""
        try:
            size = dist.get_world_size(group)
            ranks = []
            for index in range(size):
                try:
                    ranks.append(int(dist.get_global_rank(group, index)))
                except Exception:  # pylint: disable=broad-except
                    continue
        except Exception:  # pylint: disable=broad-except
            ranks = None

        try:
            backend = group._get_backend(torch.device("npu"))  # pylint: disable=protected-access
            local_rank = dist.get_rank(group)
            global_rank = dist.get_global_rank(group, local_rank)
            name = str(backend.get_hccl_comm_name(global_rank))
        except Exception:  # pylint: disable=broad-except
            name = ""
        if not name:
            name = "group:" + ",".join(str(item) for item in (ranks or []))

        result = (name, ranks)
        self._group_rank_cache[id(group)] = result  # type: ignore[assignment]
        return result

    def _next_occurrence(self, group_name: str, comm_type: str) -> int:
        key = (group_name, comm_type)
        value = self._occurrence.get(key, 0)
        self._occurrence[key] = value + 1
        return value

    def _make_collective_wrapper(self, original: Callable, api_name: str, comm_type: str):
        capture = self

        def wrapper(*args, **kwargs):
            event = capture._record_collective_start(api_name, comm_type, args, kwargs)
            result = original(*args, **kwargs)
            if event is not None:
                capture._after_collective(event, result)
            return result

        wrapper.__name__ = getattr(original, "__name__", api_name)
        wrapper.__doc__ = getattr(original, "__doc__", "")
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        return wrapper

    def _collective_payload(self, api_name: str, comm_type: str, args, kwargs) -> Tuple[Optional[int], Optional[str]]:
        """从集合通信实参里挑出"代表本次通信量的张量"，返回 (numel, dtype)。

        不同 API 的签名差异很大，这里按候选顺序试：张量、张量列表、
        ``output_tensor=`` / ``input_tensor=`` 关键字。都拿不到就返回 (None, None)，
        节点仍然进图，只是没有通信量，跨卡合并会退化为不可配对。
        """
        candidates: List[Any] = []

        def add_tensors(item):
            if item is None:
                return
            if hasattr(item, "numel"):
                candidates.append(item)
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    if hasattr(sub, "numel"):
                        candidates.append(sub)

        add_tensors(kwargs.get("tensor"))
        add_tensors(kwargs.get("tensor_list"))
        add_tensors(kwargs.get("output_tensor"))
        add_tensors(kwargs.get("input_tensor"))
        for value in args:
            add_tensors(value)

        for tensor in candidates:
            described = _describe_tensor(tensor, "payload")
            if described is not None and described.numel:
                return described.numel, described.dtype

        # barrier 之类没有张量：按 0 字节记，仍然是有效节点。
        if comm_type == "barrier":
            return 0, None
        return None, None

    def _record_collective_start(self, api_name: str, comm_type: str, args, kwargs) -> Optional[str]:
        with self._lock:
            if len(self.events) >= self.max_events:
                return None
            group = kwargs.get("group")
            if group is None:
                group = next((item for item in args if hasattr(item, "size") and hasattr(item, "rank")), None)
            group_name, ranks = self._resolve_group(group)
            numel, dtype = self._collective_payload(api_name, args, kwargs)
            occurrence = self._next_occurrence(group_name, comm_type)

            nbytes = None
            width = ir.dtype_size(dtype)
            if numel is not None and width is not None:
                nbytes = int(numel) * int(width)

            comm = ir.CommMeta(
                comm_type=comm_type,
                group_name=group_name,
                comm_name=group_name,
                count=int(numel) if numel is not None else None,
                dtype=dtype,
                nbytes=nbytes,
                participants=ranks,
                occurrence=occurrence,
            )
            deps, sources = self._take_deps(args, kwargs, result=None)
            event = ir.Event(
                eid=f"c{self._seq}",
                seq=self._seq,
                kind=ir.EVENT_COMM_COLL,
                name=f"{api_name}_{comm_type}_{occurrence}",
                deps=deps,
                dep_sources=sources,
                comm=comm,
                attrs={"api": api_name, "async": bool(kwargs.get("async_op", False))},
            )
            self._seq += 1
            self.events.append(event)
            self._touch_thread(event)
            if kwargs.get("async_op", False):
                self._pending_comm[id(event)] = event.eid
            return event.eid

    def _after_collective(self, event_eid: str, result) -> None:
        """异步 handle 记录完成点。

        ``async_op=True`` 时 Python 层返回 Work 对象，真实同步在 ``wait()``。若不把
        依赖建在 wait 点，图里就会出现"通信还没完成后续算子就开始了"这种错误顺序。
        """
        if not hasattr(result, "wait"):
            return
        capture = self

        def wait_wrapper(*args, **kwargs):
            outcome = result.wait(*args, **kwargs)
            with capture._lock:
                capture._note_comm_completion(event_eid)
            return outcome

        try:
            result.wait = wait_wrapper
        except Exception:  # pylint: disable=broad-except
            pass

    def _note_comm_completion(self, event_eid: str) -> None:
        """把"等待点"记成一个微事件，让后续算子依赖它。

        微步骤本身不进公共图（按需求忽略），但需要它作为同步证据存在，
        否则后续算子会误依赖通信的入队点。
        """
        event = ir.Event(
            eid=f"w{self._seq}",
            seq=self._seq,
            kind=ir.EVENT_COMPUTE,
            name="comm_wait",
            deps=[event_eid],
            dep_sources=[ir.DEP_COLLECTIVE],
            attrs={"is_sync_marker": True},
        )
        self._seq += 1
        self.events.append(event)
        self._touch_thread(event)

    def _make_p2p_wrapper(self, original: Callable, api_name: str, kind: str):
        capture = self

        def wrapper(*args, **kwargs):
            event_eid = capture._record_p2p(api_name, kind, args, kwargs)
            result = original(*args, **kwargs)
            if event_eid and hasattr(result, "wait"):
                capture._after_collective(event_eid, result)
            return result

        wrapper.__name__ = getattr(original, "__name__", api_name)
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        return wrapper

    def _make_batch_p2p_wrapper(self, original: Callable):
        capture = self

        def wrapper(ops, *args, **kwargs):
            eids = []
            for op in ops or []:
                kind = "p2p_send" if "isend" in type(op).__name__.lower() or getattr(op, "op", None) else "p2p_recv"
                eids.append(capture._record_p2p("batch_isend_irecv", kind, (op,), {}))
            result = original(ops, *args, **kwargs)
            if hasattr(result, "__iter__"):
                for item in result:
                    if hasattr(item, "wait"):
                        eid = eids[0] if eids else None
                        if eid:
                            capture._after_collective(eid, item)
                        break
            return result

        wrapper.__name__ = "batch_isend_irecv"
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        return wrapper

    def _record_p2p(self, api_name: str, kind: str, args, kwargs) -> Optional[str]:
        with self._lock:
            if len(self.events) >= self.max_events:
                return None
            peer = kwargs.get("dst") if kind == "p2p_send" else kwargs.get("src")
            if peer is None:
                for value in args:
                    if isinstance(value, int):
                        peer = value
                        break
            tensor = kwargs.get("tensor")
            if tensor is None:
                tensor = next((item for item in args if hasattr(item, "numel")), None)
            described = _describe_tensor(tensor, "payload") if tensor is not None else None
            nbytes = described.nbytes() if described is not None else None
            comm = ir.CommMeta(
                comm_type="send" if kind == "p2p_send" else "recv",
                src_rank=self.rank if kind == "p2p_send" else peer,
                dst_rank=peer if kind == "p2p_send" else self.rank,
                tag=kwargs.get("tag"),
                count=described.numel if described is not None else None,
                dtype=described.dtype if described is not None else None,
                nbytes=nbytes,
            )
            deps, sources = self._take_deps(args, kwargs, result=None)
            event = ir.Event(
                eid=f"p{self._seq}",
                seq=self._seq,
                kind=ir.EVENT_P2P_SEND if kind == "p2p_send" else ir.EVENT_P2P_RECV,
                name=f"{api_name}_{kind}",
                deps=deps,
                dep_sources=sources,
                comm=comm,
                attrs={"api": api_name},
            )
            self._seq += 1
            self.events.append(event)
            self._touch_thread(event)
            return event.eid

    # -- 融合算子钩子 ----------------------------------------------------
    def _install_fused_hooks(self) -> bool:
        """包装规则表里登记过的融合算子入口。

        目标有两类：
            - ``torch_npu`` 上的 ``npu_*`` 融合算子；
            - MindSpeed ``coc_ops`` 上的 CoC 方法。

        平台不存在时静默跳过（CPU 单测环境会走到这里）。
        """
        installed = 0
        targets: List[Any] = []
        try:
            import torch_npu  # noqa: F401  pylint: disable=import-outside-toplevel

            targets.append(torch_npu)
        except Exception:  # pylint: disable=broad-except
            self._notes.append("torch_npu 不可用，跳过 npu_* 融合算子采集")

        for module_name, attr in (
            ("mindspeed.ops.lcal_functional", "coc_ops"),
            ("mindspeed.ops", "coc_ops"),
        ):
            try:
                module = __import__(module_name, fromlist=[attr])
                instance = getattr(module, attr, None)
                if instance is not None:
                    targets.append(instance)
            except Exception:  # pylint: disable=broad-except
                continue

        if not targets:
            self._notes.append("没有找到可包装的融合算子入口")

        for target in targets:
            for name in list(fused_ops.FUSED_RULES):
                if not hasattr(target, name):
                    continue
                original = getattr(target, name)
                if not callable(original):
                    continue
                self._originals.append((target, name, original))
                try:
                    setattr(target, name, self._make_fused_wrapper(original, name))
                    installed += 1
                except Exception:  # pylint: disable=broad-except
                    self._originals.pop()
        return installed > 0

    def _make_fused_wrapper(self, original: Callable, op_name: str):
        capture = self

        def wrapper(*args, **kwargs):
            eid = capture._record_fused_start(op_name, original, args, kwargs)
            result = original(*args, **kwargs)
            if eid:
                capture._record_fused_end(eid, result)
            return result

        wrapper.__name__ = getattr(original, "__name__", op_name)
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        return wrapper

    def _bind_arguments(self, original: Callable, args, kwargs) -> Dict[str, Any]:
        """把位置参数按签名绑定成名字，避免依赖参数顺序。"""
        try:
            import inspect  # pylint: disable=import-outside-toplevel

            signature = inspect.signature(original)
            bound = signature.bind_partial(*args, **kwargs)
            return dict(bound.arguments)
        except Exception:  # pylint: disable=broad-except
            named = dict(kwargs)
            for index, value in enumerate(args):
                named[f"arg{index}"] = value
            return named

    def _record_fused_start(self, op_name: str, original: Callable, args, kwargs) -> Optional[str]:
        with self._lock:
            if len(self.events) >= self.max_events:
                return None
            bound = self._bind_arguments(original, args, kwargs)
            arg_values: Dict[str, fused_ops.ArgValue] = {}
            for name, value in bound.items():
                described = _arg_value(value, name)
                if described is not None:
                    arg_values[name] = described

            rule = fused_ops.rule_for(op_name) or {}
            group_name = ""
            group_from = str(rule.get("group_from", ""))
            raw_group = bound.get(group_from)
            if raw_group is not None and not isinstance(raw_group, str):
                # comm_domain 由调用点算出；这里只把可读表示放进 context。
                group_name = str(raw_group)
            elif isinstance(raw_group, str):
                group_name = raw_group
            if not group_name:
                group_name = str(bound.get("hcomm_info") or bound.get("hcom") or "")

            occurrence = self._next_occurrence(group_name, str(rule.get("comm_type", "")))
            context = {
                "comm_name": group_name,
                group_from: group_name,
                "rank": self.rank,
            }
            spec = fused_ops.build_spec(
                fused_ops.BoundArgs(op=op_name, args=arg_values, context=context),
                occurrence=occurrence,
                resolve_group=None,
            )

            deps, sources = self._take_deps(args, kwargs, result=None)
            event = ir.Event(
                eid=f"f{self._seq}",
                seq=self._seq,
                kind=ir.EVENT_FUSED,
                name=op_name,
                deps=deps,
                dep_sources=sources,
                fused=spec,
                attrs={
                    "op": op_name,
                    "arg_names": sorted(arg_values),
                    "group_name": group_name,
                },
            )
            self._seq += 1
            self.events.append(event)
            self._touch_thread(event)
            return event.eid

    def _record_fused_end(self, event_eid: str, result) -> None:
        """记录融合算子的输出张量，让后续算子依赖它。

        输出必须登记为本事件写入的张量，否则下一个算子找不到"谁生产了它"，
        数据依赖链会断。
        """
        with self._lock:
            keys = _tensor_keys(result, self._torch)
            if not keys:
                return
            for key in keys:
                self._last_writer[key] = event_eid
            self._written_keys.setdefault(event_eid, []).extend(keys)

    # -- Dispatch 钩子 ---------------------------------------------------
    def _install_dispatch_mode(self) -> bool:
        try:
            from torch.utils._python_dispatch import TorchDispatchMode  # pylint: disable=import-outside-toplevel
        except Exception as error:  # pylint: disable=broad-except
            self._notes.append(f"TorchDispatchMode 不可用，跳过计算算子采集: {error}")
            self._degraded = True
            return False

        capture = self

        class _CaptureMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=arguments-differ
                kwargs = kwargs or {}
                eid = capture._on_dispatch_start(func, args, kwargs)
                result = func(*args, **kwargs)
                # 输出张量必须在调用**之后**登记：多数 Aten 算子不写 out=，而是返回新张量。
                # 若不登记返回值，"谁生产了这个张量"就永远查不到，数据依赖会全部退化成
                # 线程顺序依赖——那正是本方案要避免的。
                if eid is not None:
                    capture._on_dispatch_end(eid, result)
                return result

        try:
            self._dispatch_handle = _CaptureMode()
            self._dispatch_handle.__enter__()
        except Exception as error:  # pylint: disable=broad-except
            self._notes.append(f"无法进入 TorchDispatchMode: {error}")
            self._degraded = True
            return False
        return True

    def _on_dispatch_start(self, func, args, kwargs) -> Optional[str]:
        """dispatch 回调（调用前）：记录算子并返回事件 id，供调用后登记输出。"""
        with self._lock:
            if len(self.events) >= self.max_events:
                return None
            name = str(getattr(func, "__name__", func))
            if name.startswith("_"):
                base_name = name
            else:
                try:
                    base_name = str(func.overloadpacket)  # pylint: disable=no-member
                except Exception:  # pylint: disable=broad-except
                    base_name = name

            # functional collectives 在 Aten 层可见，按通信节点记录。
            if name.startswith(_FUNCTIONAL_COLLECTIVE_PREFIX):
                self._record_functional_collective(name, args, kwargs)
                return None

            deps, sources = self._take_deps(args, kwargs, result=None)
            event = ir.Event(
                eid=f"o{self._seq}",
                seq=self._seq,
                kind=ir.EVENT_COMPUTE,
                name=base_name,
                deps=deps,
                dep_sources=sources,
                attrs={"overload": name},
            )
            self._seq += 1
            self.events.append(event)
            self._touch_thread(event)
            return event.eid

    def _on_dispatch_end(self, event_eid: str, result) -> None:
        """dispatch 回调（调用后）：把返回张量登记为本次事件写入的产物。"""
        keys = _tensor_keys(result, self._torch)
        if not keys:
            return
        with self._lock:
            for key in keys:
                self._last_writer[key] = event_eid
            self._written_keys.setdefault(event_eid, []).extend(keys)

    def _record_functional_collective(self, name: str, args, kwargs) -> None:
        comm_type = "all_reduce"
        for candidate, mapped in (
            ("all_gather", "all_gather"),
            ("reduce_scatter", "reduce_scatter"),
            ("all_to_all", "all_to_all"),
            ("broadcast", "broadcast"),
            ("all_reduce", "all_reduce"),
        ):
            if candidate in name:
                comm_type = mapped
                break
        group_name, ranks = self._resolve_group(None)
        described = None
        for value in args:
            described = _arg_value(value, "payload")
            if described is not None:
                break
        occurrence = self._next_occurrence(group_name, comm_type)
        comm = ir.CommMeta(
            comm_type=comm_type,
            group_name=group_name,
            comm_name=group_name,
            count=described.numel if described is not None else None,
            dtype=described.dtype if described is not None else None,
            nbytes=described.nbytes() if described is not None else None,
            participants=ranks,
            occurrence=occurrence,
        )
        deps, sources = self._take_deps(args, kwargs, result=None)
        event = ir.Event(
            eid=f"c{self._seq}",
            seq=self._seq,
            kind=ir.EVENT_COMM_COLL,
            name=f"functional_{comm_type}_{occurrence}",
            deps=deps,
            dep_sources=sources,
            comm=comm,
            attrs={"api": name, "functional": True},
        )
        self._seq += 1
        self.events.append(event)
        self._touch_thread(event)

    # -- 依赖计算 --------------------------------------------------------
    def _take_deps(self, args, kwargs, result) -> Tuple[List[str], List[str]]:
        """推导本次调用的依赖。

        规则（按强度排序）：
            1. **真实数据依赖**：入参张量的 ``(id, version)`` 上一次被哪个事件写过。
            2. **同线程顺序**：同一 Python 线程上的上一个事件。这保证控制流顺序，
               但比数据依赖弱，因此单独标注 ``sequence``。
        """
        deps: List[str] = []
        sources: List[str] = []
        if self.capture_tensor_keys and self._last_writer:
            keys = _tensor_keys(args, self._torch) + _tensor_keys(kwargs, self._torch)
            seen = set()
            for key in keys:
                if key in seen:
                    continue
                seen.add(key)
                writer = self._last_writer.get(key)
                if writer is not None and writer not in deps:
                    deps.append(writer)
                    sources.append(ir.DEP_TENSOR)
        thread_last = self._thread_last.get(threading.get_ident())
        if thread_last is not None and thread_last not in deps:
            deps.append(thread_last)
            sources.append(ir.DEP_SEQUENCE)
        return deps, sources

    def _touch_thread(self, event: ir.Event) -> None:
        self._thread_last[threading.get_ident()] = event.eid

    # -- 导出 ------------------------------------------------------------
    def dump(self, output_dir: str) -> Dict[str, Any]:
        """把采集结果落盘为 ``events_rank<N>.json``，返回摘要。

        写出放在子进程里做（见 ``export``），这里只负责生成 JSON。
        为控制体积，只保留图中真正需要的字段。
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"events_rank{self.rank}.json")
        meta = {
            "rank": self.rank,
            "world_size": self.world_size,
            "base_ts_us": self._base_ts_us,
            "window_seconds": (
                None
                if self._window_start is None or self._window_end is None
                else self._window_end - self._window_start
            ),
            "degraded": self._degraded,
            "notes": self.notes(),
            "max_events_hit": len(self.events) >= self.max_events,
        }
        ir.dump_events(self.events, path, meta)
        return {
            "rank": self.rank,
            "path": path,
            "events": len(self.events),
            "degraded": self._degraded,
            "notes": self.notes(),
        }

    # -- 上下文管理 ------------------------------------------------------
    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.uninstall()
        return False


@contextlib.contextmanager
def capture_scope(rank: int = 0, world_size: int = 1, **kwargs):
    """便捷上下文管理器；退出时保证钩子被卸下，即使中间抛异常。"""
    dag_capture = DagCapture(rank=rank, world_size=world_size, **kwargs)
    dag_capture.install()
    try:
        yield dag_capture
    finally:
        dag_capture.uninstall()
