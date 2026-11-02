# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Ascend 通算 DAG（Chakra execution trace）导出工具。

本包把"运行时记录的结构"和"profiling 测得的时长"合并成一张计算+通信有向无环图，
并以 MLCommons Chakra 的 execution trace protobuf 格式输出。

模块划分：

===============  ==========================================================
模块             职责
===============  ==========================================================
``ir``           中间表示：事件记录、DAG 节点、边、以及序列化。
``fused_ops``    融合算子（MC2 / CoC / ascend_turbo）展开规则表。
``graph``        由事件记录构建单卡 DAG，多卡 collective 合并，环检测。
``timing``       解析 ``trace_view.json``，并把实测时长 join 到 DAG 上。
``writer``       写 Chakra protobuf（``.et``）与合并图 JSON。
``capture``      训练期运行时采集（唯一依赖 torch 的模块，延迟导入）。
``export``       编排：定位 trace、起子进程、汇总产物。
``cli``          离线命令行入口。
===============  ==========================================================

导入约定：``import mindspeed_llm.tools.chakra_dag`` 不会导入 torch，也不会导入
chakra。只有调用到 ``capture`` 或 ``writer`` 的内部函数时才会真正加载它们。
"""

__all__ = [
    "ir",
    "fused_ops",
    "graph",
    "timing",
    "writer",
    "capture",
    "export",
    "cli",
]
