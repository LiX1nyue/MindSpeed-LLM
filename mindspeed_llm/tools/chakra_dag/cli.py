# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""离线命令行入口：``python -m mindspeed_llm.tools.chakra_dag.cli``。

三种用法：

1. **补全已有采集结果**（默认，最常用）::

       python -m mindspeed_llm.tools.chakra_dag.cli \\
           --output ./profile_dir/chakra_dag \\
           --format both

   自动发现 ``./profile_dir/chakra_dag/events_rank*.json``，建图并写产物。

2. **只用 trace 兜底**（没有运行期采集时）::

       python -m mindspeed_llm.tools.chakra_dag.cli \\
           --profile-path ./profile_dir --source trace \\
           --output ./dag_from_trace

   这时结构由 trace 时间顺序给出，跨卡 collective 配对不可靠，
   报告里会标 ``structure="trace_only"``。

3. **自检**：``--check`` 只做读回校验，不重新导出。

命令行绝不因为 chakra 缺失而失败：缺 chakra 时跳过 ``.et``，仍然写 JSON 与报告。
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import export, ir, writer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mindspeed_llm.tools.chakra_dag.cli",
        description="把运行时采集的结构 + profiling 实测时长合成 Chakra 通算 DAG。",
    )
    parser.add_argument(
        "--events",
        nargs="*",
        default=None,
        help="运行期采集的 events_rank*.json 路径列表；缺省时在 --output 目录里自动发现。",
    )
    parser.add_argument(
        "--profile-path",
        default=None,
        help="profiling 根目录（含 trace_view.json 的那棵树的根）。用于 --source trace 或校验。",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "capture", "trace"),
        default="auto",
        help="结构来源：capture=运行期采集（推荐）；trace=从 trace 兜底；auto=有采集用采集。",
    )
    parser.add_argument("--output", required=True, help="产物输出目录。")
    parser.add_argument(
        "--format",
        choices=("protobuf", "json", "both"),
        default="both",
        help="输出格式；json 不需要安装 chakra。",
    )
    parser.add_argument(
        "--fused-mode",
        choices=(ir.SPLIT_MEASURED, ir.SPLIT_ESTIMATE, ir.SPLIT_MERGED),
        default=ir.SPLIT_ESTIMATE,
        help="融合算子内部时长拆分策略，详见 fused_ops 模块文档。",
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=None,
        help="估算融合通信段时使用的带宽（GB/s）。缺省时用 trace 里同组的实测带宽。",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="不做跨卡合并，只输出每个 rank 单卡图。",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="有环等严重问题时只记录不报错（默认 strict 直接失败）。",
    )
    parser.add_argument(
        "--require-participants",
        action="store_true",
        help="要求每个 collective 都有 participants，缺失记为问题。",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只读回并校验已有 .et，不重新导出。",
    )
    return parser


def _discover_events(output_dir: str, explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return [path for path in explicit if os.path.exists(path)]
    if output_dir and os.path.isdir(output_dir):
        return export.collect_rank_events(output_dir)
    return []


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    output_dir = os.path.abspath(args.output)

    if args.check:
        return _check(output_dir)

    report: Dict[str, Any] = {"output_dir": output_dir, "args": vars(args), "warnings": [], "errors": []}

    event_paths = _discover_events(output_dir, args.events)
    use_capture = args.source in ("auto", "capture") and bool(event_paths)
    if args.source == "capture" and not event_paths:
        report["errors"].append(f"--source capture 但没有找到 events_rank*.json（已查找 {output_dir}）")
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return report

    graphs: List[ir.RankGraph] = []
    if use_capture:
        graphs, build_report = export.build_graphs_from_events(
            event_paths,
            fused_mode=args.fused_mode,
            default_bandwidth_gbps=args.bandwidth_gbps,
        )
        # 结构来自运行时采集，时长来自 profiling —— 两件事分开做。
        export.attach_trace_timings(
            graphs,
            args.profile_path,
            build_report,
            bandwidth_gbps=args.bandwidth_gbps,
            fused_mode=args.fused_mode,
        )
    elif args.profile_path:
        trace_paths = export.find_trace_files(args.profile_path)
        if not trace_paths:
            report["errors"].append(f"在 {args.profile_path} 下没找到 trace_view.json")
            print(json.dumps(report, ensure_ascii=False, indent=1))
            return report
        graphs, build_report = export.build_graphs_from_traces(
            trace_paths,
            output_dir,
            fused_mode=args.fused_mode,
            default_bandwidth_gbps=args.bandwidth_gbps,
            require_participants=args.require_participants,
        )
    else:
        report["errors"].append(
            "既没有 events_rank*.json，也没有给 --profile-path，无法建图。"
            "训练时请加 --chakra-dag-export 先做一次运行期采集，"
            "或用 --profile-path 指向 profiling 目录做 trace 兜底。"
        )
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return report

    report["build"] = build_report
    # 把建图阶段收集的 warning/timing 合并进最终报告，否则 export_report.json 里看不到
    # "哪个 rank 没找到 trace""多少节点时长未知"这类关键信息。
    for key in ("warnings", "timing", "errors"):
        if key not in build_report:
            continue
        target = report.setdefault(key, [])
        for item in build_report[key]:
            if item not in target:
                target.append(item)
    if not graphs:
        report["errors"].append("没有构建出任何 rank 图，请检查输入。")
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return report

    export.write_manifest(
        graphs,
        output_dir,
        extra={
            "profile_path": args.profile_path,
            "source": build_report.get("structure"),
            "fused_mode": args.fused_mode,
        },
    )
    report = export.write_outputs(
        graphs,
        output_dir,
        report,
        export_format=args.format,
        merge_multi_rank=not args.no_merge,
        metadata_extra={"world_size": len(graphs)},
        strict=not args.lenient,
    )

    summary = {
        "output_dir": output_dir,
        "ranks": [item.rank for item in graphs],
        "structure": build_report.get("structure"),
        "warnings": report.get("warnings", []),
        "errors": report.get("errors", []),
        "merge": report.get("merge", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return report


def _check(output_dir: str) -> Dict[str, Any]:
    """读回 ``.et`` 做自检：元数据版本、节点数、依赖是否指向已存在的节点。"""
    status = writer.describe_chakra_status()
    result: Dict[str, Any] = {"output_dir": output_dir, "chakra": status, "files": []}
    if not status["available"]:
        result["error"] = "chakra 不可用，无法读回 .et"
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return result

    for name in sorted(os.listdir(output_dir)):
        if not name.endswith(".et"):
            continue
        path = os.path.join(output_dir, name)
        metadata, nodes = writer.read_chakra_file(path)
        ids = {node.id for node in nodes}
        dangling = []
        for node in nodes:
            for dep in node.data_deps:
                if dep not in ids:
                    dangling.append((node.id, dep))
        cycles_free = not dangling
        result["files"].append(
            {
                "path": path,
                "version": getattr(metadata, "version", None),
                "nodes": len(nodes),
                "dangling_deps": len(dangling),
                "dangling_sample": dangling[:10],
                "types": _type_counts(nodes),
                "ok": cycles_free and len(dangling) == 0,
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def _type_counts(nodes) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in nodes:
        key = str(node.type)
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    main(sys.argv[1:])
