# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""训练侧接入点：把运行时采集挂到 profiling 生命周期上。

## 为什么要单独一个模块

训练代码里只需要插入很少几行。把这些行背后的判断（是否启用、哪个 rank 参与、
何时 install/uninstall、失败怎么降级）收在这里：

- ``training.py`` / ``profiler.py`` 的改动可以小到 3 行，审阅成本低；
- 采集策略变化时不必再碰训练主流程；
- 这些判断可以脱离训练框架直接单元测试。

## 生命周期（顺序很关键）

::

    prof.start()            现有代码
    dag.begin(args, rank)   ← 新增：安装钩子，开始记录结构
      ... 训练 step ...
    dag.end(args, rank)     ← 新增：卸钩子并写 events_rank<N>.json
    prof.stop()             现有代码；profiling 产物在这里落盘
    dag.export(args)        ← 新增：trace 已存在，起子进程补时长 + 写 .et

为什么 ``end`` 在 ``prof.stop()`` **之前**：采集记录不应覆盖 profiler 自己的收尾
FLUSH 与数据导出阶段，否则会混进大量与训练无关的 I/O 算子。

为什么 ``export`` 在 ``prof.stop()`` **之后**：``trace_view.json`` 那时才由
``tensorboard_trace_handler`` 写出。

## 失败策略

**任何一步失败都不能中断训练。** 所有入口返回状态字典、不抛异常；采集装不上就
当作没开启，导出失败只是少一份分析产物。
"""

import os
from typing import Any, Dict, Optional

from . import capture as capture_mod
from . import export as export_mod

#: rank -> 采集器实例。训练进程生命周期内有效。
_ACTIVE: Dict[int, "capture_mod.DagCapture"] = {}


def begin(args, rank: int) -> Optional["capture_mod.DagCapture"]:
    """训练循环开始前调用；未启用或安装失败时返回 None。

    幂等：同一 rank 重复调用返回已有实例。
    """
    if not getattr(args, "chakra_dag_export", False):
        return None
    if rank in _ACTIVE:
        return _ACTIVE[rank]
    try:
        recorder = capture_mod.DagCapture(
            rank=rank,
            world_size=int(getattr(args, "world_size", 1) or 1),
            max_events=int(getattr(args, "chakra_dag_max_events", 2_000_000)),
        )
        installed = recorder.install()
        if not installed and recorder.degraded:
            # 一个钩子都没装上（典型场景：非 NPU 环境、torch 不可用）。
            # 此时不登记，避免 status() 误报"正在采集"。
            return None
        _ACTIVE[rank] = recorder
        return recorder
    except Exception:  # pylint: disable=broad-except
        # 采集是可选功能，装不上就当作没开。
        return None


def end(args, rank: int, output_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """卸下钩子并把采集结果落盘。应在 ``prof.stop()`` **之前** 调用。

    只有在采集确实记录到事件时才写文件；否则不落盘，避免用一个空文件覆盖掉
    上一次运行留下的有效采集结果（调试时反复跑同一目录很容易踩到）。
    """
    recorder = _ACTIVE.pop(rank, None)
    if recorder is None:
        return None
    try:
        recorder.uninstall()
    except Exception:  # pylint: disable=broad-except
        pass
    if not recorder.events:
        return {
            "rank": rank,
            "events": 0,
            "skipped": True,
            "reason": "没有采集到任何事件（钩子未安装成功或被过滤），未写出文件",
            "notes": recorder.notes(),
        }
    try:
        return recorder.dump(output_dir or default_output_dir(args))
    except Exception as error:  # pylint: disable=broad-except
        return {"ok": False, "error": str(error)}


def export(args) -> Optional[Dict[str, Any]]:
    """``prof.stop()`` **之后** 调用：起子进程补时长并写产物。

    只有 rank0 执行，避免多 rank 同时写同一目录。前提是各 rank 能看到同一个
    ``events_rank*.json`` 目录（共享文件系统）；否则只导出本 rank，报告里
    会提示需要人工汇总。
    """
    if not getattr(args, "chakra_dag_export", False):
        return None
    rank = int(getattr(args, "rank", 0) or 0)
    if rank != 0:
        return None
    try:
        return export_mod.maybe_export(args, rank, int(getattr(args, "world_size", 1) or 1))
    except Exception as error:  # pylint: disable=broad-except
        return {"ok": False, "error": str(error)}


def default_output_dir(args) -> str:
    """默认输出目录：profiling 根目录下的 ``chakra_dag/``。"""
    root = getattr(args, "profile_save_path", "./profile_dir") or "./profile_dir"
    return os.path.join(root, "chakra_dag")


def status() -> Dict[str, Any]:
    """当前进程的采集状态，便于日志自检。"""
    return {"active_ranks": sorted(_ACTIVE)}
