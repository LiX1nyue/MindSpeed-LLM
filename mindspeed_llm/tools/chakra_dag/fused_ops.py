# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""融合算子（MC2 / CoC / ascend_turbo）展开规则表。

## 问题

MindSpeed 大量使用把通信和计算编译进同一个 NPU kernel 的融合算子，例如
``torch_npu.npu_all_gather_base_mm`` 内部同时做 all-gather 和 matmul。在 Aten 层
它只是一个不透明的算子：

- 如果原样记成一个 ``COMP_NODE``，图里就悄悄少了一次集合通信；
- 如果只记成 ``COMM_COLL_NODE``，图里就悄悄少了一次矩阵乘。

两种都是静默失真，不能接受。所以必须在这一层把它**展开**成两个节点。

## 为什么展开是可行的

不需要进入 kernel 内部：本仓库对这些算子的**调用点**已经把参数写清楚了。
例如 ``mindspeed_ops/lcal_functional.py`` 里：

```python
def matmul_all_reduce(self, input1, input2, output, bias=None):
    tp_size = mpu.get_tensor_model_parallel_world_size()
    comm_domain = str(device // tp_size)
    CoCOperations.mindspeed_ops.matmul_all_reduce(input1, input2, bias, output,
                                                 rank, tp_size, comm_domain)
```

``comm_domain`` 就是参与组，``input1/input2`` 的 shape 就是计算量，
输出 shape 和 dtype 就是通信量。这些信息在运行时是**已知事实**，不是猜测。

## 展开后的拓扑：兄弟节点，不是串行链

融合的意义在于通信与计算重叠。若展开成 ``comm -> comp`` 串行，会：

1. 强加一条并不存在的依赖；
2. 让仿真总时长变成 ``t_comm + t_compute``，明显大于实测融合耗时。

因此展开后的两个节点共享同一组前驱与后继，时间区间都落在实测融合包络内。
具体重合多少由 ``SPLIT_*`` 策略决定，并在节点属性里标注 ``time_split``。

## 本模块只做纯字符串/数值处理

为了让规则表能在没有 torch、没有 NPU 的本机做单元测试，这里的输入是已经
规范化过的 ``BoundArgs``：参数名 -> ``ArgValue``。把 torch 张量翻译成
``ArgValue`` 的工作在 ``capture`` 模块完成。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import ir

# ---------------------------------------------------------------------------
# 融合模式：op 名 -> (展开顺序, 通信类型, 通信量取自哪个参数)
# ---------------------------------------------------------------------------
# ops 里的元素只有两种：具体算子名（进图作为计算节点）和 collective 名
# （进图作为通信节点）。collective 名必须出现在 ir.COMM_TYPES 里。

#: 全部融合展开规则。键是算子/方法名，值是规则字典。
FUSED_RULES: Dict[str, Dict[str, Any]] = {
    # ---- MindSpeed CoC (mindspeed.ops.lcal_functional.CoCOperations) ----
    # 行并行线性层的 "matmul + all_reduce"，bias 可缺省。
    "matmul_all_reduce": {
        "ops": ("matmul", "all_reduce"),
        "comm_type": "all_reduce",
        "size_from": "output",  # all_reduce 通信量取输出张量
        "order": "comp_first",
        "group_from": "comm_domain",
    },
    # 列并行线性层的 "all_gather + matmul"，gather 后做 matmul。
    "all_gather_matmul": {
        "ops": ("all_gather", "matmul"),
        "comm_type": "all_gather",
        "size_from": "input1",  # all_gather 的每个 rank 发送量 = 本地输入
        "order": "comm_first",
        "group_from": "comm_domain",
    },
    "all_gather_matmul_v2": {
        "ops": ("all_gather", "matmul"),
        "comm_type": "all_gather",
        "size_from": "input1",
        "order": "comm_first",
        "group_from": "comm_domain",
    },
    "matmul_reduce_scatter": {
        "ops": ("matmul", "reduce_scatter"),
        "comm_type": "reduce_scatter",
        "size_from": "input1",
        "order": "comp_first",
        "group_from": "comm_domain",
    },
    # TP2D：AG + MM + RS 三段融合，通信取第一段（对图结构足够）。
    "all_gather_matmul_reduce_scatter": {
        "ops": ("all_gather", "matmul", "reduce_scatter"),
        "comm_type": "all_gather",
        "size_from": "input1",
        "order": "comm_first",
        "group_from": "comm_domain",
        "extra_comm": ("reduce_scatter", "input1"),
    },
    # ---- torch_npu MC2 融合算子 ----
    "npu_all_gather_base_mm": {
        "ops": ("all_gather", "matmul"),
        "comm_type": "all_gather",
        "size_from": "x",
        "order": "comm_first",
        "group_from": "hcomm_info",
    },
    "npu_mm_reduce_scatter_base": {
        "ops": ("matmul", "reduce_scatter"),
        "comm_type": "reduce_scatter",
        "size_from": "x",
        "order": "comp_first",
        "group_from": "hcomm_info",
    },
    # MoE MC2：all-to-all(v) 与 grouped matmul 融合。
    "npu_alltoallv_gmm": {
        "ops": ("all_to_all", "grouped_matmul"),
        "comm_type": "all_to_all",
        "size_from": "gmm_x",
        "order": "comm_first",
        "group_from": "hcom",
    },
    "npu_grouped_mat_mul_all_reduce": {
        "ops": ("grouped_matmul", "all_reduce"),
        "comm_type": "all_reduce",
        "size_from": "x",
        "order": "comp_first",
        "group_from": "hcom",
    },
    # 双段融合：BMM -> RS -> A2A / A2A -> AG -> BMM。
    "npu_bmm_reduce_scatter_all_to_all": {
        "ops": ("matmul", "reduce_scatter", "all_to_all"),
        "comm_type": "reduce_scatter",
        "size_from": "x",
        "order": "comp_first",
        "group_from": "hcom",
    },
    "npu_all_to_all_all_gather_bmm": {
        "ops": ("all_to_all", "all_gather", "matmul"),
        "comm_type": "all_to_all",
        "size_from": "x",
        "order": "comm_first",
        "group_from": "hcom",
    },
    # 三段融合：MM -> AR -> RMSNorm。
    "npu_mm_all_reduce_add_rms_norm": {
        "ops": ("matmul", "all_reduce", "rms_norm"),
        "comm_type": "all_reduce",
        "size_from": "x1",
        "order": "comp_first",
        "group_from": "hcom",
    },
    # 单 collective 的纯通信封装，也登记进来以便统一走展开路径。
    "npu_all_gather_base_mm_v2": {
        "ops": ("all_gather", "matmul"),
        "comm_type": "all_gather",
        "size_from": "x",
        "order": "comm_first",
        "group_from": "hcomm_info",
    },
}

#: 名字里带这些片段、但**不该**被当作融合通信算子的导出符号（避免误报）。
NOT_FUSED_PATTERNS: Tuple[str, ...] = (
    "hccl_comm_name",
    "get_hccl",
    "comm_domain",
    "comm_group",
    "comm_utils",
    "comm_config",
)

#: 判断"这个名字看起来应该有通信"的关键字，用于字典完整性检查。
COMM_HINT_PATTERN = "comm"

#: 判断"这个名字看起来应该两段都有"的关键字。
FUSED_HINT_PATTERNS: Tuple[str, ...] = (
    "all_gather",
    "allgather",
    "reduce_scatter",
    "alltoall",
    "all_to_all",
    "mm_all_reduce",
    "matmul_all_reduce",
    "bmm_reduce_scatter",
)


@dataclass
class ArgValue:
    """规范化后的单个实参。

    ``value`` 是原始值的可 JSON 化表示（这里只需要 shape/dtype/numel），
    ``numel`` 和 ``dtype`` 用于计算通信字节数。
    """

    name: str
    numel: Optional[int] = None
    dtype: Optional[str] = None
    shape: Optional[Sequence[int]] = None
    # 字符串参数（group 名、hcomm 名）原样保留。
    text: Optional[str] = None

    def nbytes(self) -> Optional[int]:
        width = ir.dtype_size(self.dtype)
        if self.numel is None or width is None:
            return None
        return int(self.numel) * width


@dataclass
class BoundArgs:
    """一次融合算子调用的已绑定实参集合。

    ``args`` 是合并了位置参数与关键字参数的 dict。位置参数的名字由
    ``capture`` 模块按目标函数的签名绑定得出，因此不要在这里做位置推断——
    签名可能随版本变化，绑定点更清楚。
    """

    op: str
    args: Dict[str, ArgValue] = field(default_factory=dict)
    # 调用点的额外上下文，例如 comm_domain / rank / world_size。
    context: Dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Optional[ArgValue]:
        return self.args.get(name)

    def first_of(self, names: Sequence[str]) -> Optional[ArgValue]:
        for name in names:
            value = self.args.get(name)
            if value is not None:
                return value
        return None


def is_fused_op(op_name: str) -> bool:
    """名字是否命中展开规则表。"""
    base = op_name.rsplit(".", 1)[-1]
    return base in FUSED_RULES


def rule_for(op_name: str) -> Optional[Dict[str, Any]]:
    base = op_name.rsplit(".", 1)[-1]
    return FUSED_RULES.get(base)


def build_spec(bound: BoundArgs, occurrence: int = 0, resolve_group=None) -> Optional[ir.FusedSpec]:
    """把一次融合算子调用翻译成 ``FusedSpec``。

    输入：
        bound：已绑定实参。
        occurrence：该 (group, comm_type) 下的序号，跨卡配对的确定性主键。
        resolve_group：可选的组名 -> global rank 列表解析函数，
            签名 ``(group_name) -> Optional[List[int]]``。

    输出：
        ``FusedSpec``；当该算子没有对应规则时返回 None（调用方应退化为普通
        计算节点，并在报告里记一次"未登记的疑似融合算子"）。
    """
    rule = rule_for(bound.op)
    if rule is None:
        return None

    ops = [str(item) for item in rule["ops"]]
    comm_type = str(rule["comm_type"])
    size_arg = bound.get(str(rule.get("size_from", "")))
    nbytes = size_arg.nbytes() if size_arg is not None else None
    count = int(size_arg.numel) if size_arg is not None and size_arg.numel is not None else None
    dtype = size_arg.dtype if size_arg is not None else None

    group_name = ""
    group_from = str(rule.get("group_from", ""))
    if group_from:
        group_arg = bound.get(group_from)
        if group_arg is not None and group_arg.text:
            group_name = group_arg.text
        else:
            # comm_domain 等可能只出现在 context 里（调用点计算得出）。
            raw = bound.context.get(group_from)
            if raw is not None:
                group_name = str(raw)

    participants = None
    if resolve_group is not None and group_name:
        participants = resolve_group(group_name)

    order = str(rule.get("order", "comm_first"))
    return ir.FusedSpec(
        ops=ops,
        comm_type=comm_type,
        group_name=group_name,
        comm_name=str(bound.context.get("comm_name", "") or ""),
        count=count,
        dtype=dtype,
        nbytes=nbytes,
        participants=participants,
        occurrence=occurrence,
        order=order,
    )


def unresolved_comm_candidates(op_names: Sequence[str]) -> List[str]:
    """找出"名字暗示有通信、但规则表里没有"的算子名。

    这个函数给单元测试用。公司服务器上新增 MindSpeed/ATB 版本后，若引入了新的
    融合算子而没人补规则，这个检查会失败，从而避免静默丢失通信节点。
    """
    missing: List[str] = []
    for raw in op_names:
        name = raw.rsplit(".", 1)[-1]
        if name in FUSED_RULES:
            continue
        lowered = name.lower()
        if any(pattern in lowered for pattern in NOT_FUSED_PATTERNS):
            continue
        if not any(hint in lowered for hint in FUSED_HINT_PATTERNS):
            continue
        missing.append(name)
    return missing
