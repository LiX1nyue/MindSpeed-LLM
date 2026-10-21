# Chakra 执行轨迹转换

本文档介绍如何把昇腾 Profiling 采集到的 trace 文件转换成 Chakra execution trace，以便使用 Chakra 生态的可视化和分析工具。采集 trace 文件本身的方法请参考[性能数据采集](profiling.md)。

想先通俗理解转换器如何从 trace 生成计算+通信依赖图（通算 DAG）的读者，可以先阅读[通算 DAG 生成逻辑通俗说明](chakra_dag_explained.md)，预计阅读约 15 分钟。

## 概述

Chakra 是 MLCommons 定义的机器学习工作负载执行轨迹标准，使用 protobuf 描述带有依赖关系的计算与通信节点。MindSpeed LLM 提供离线转换脚本[chakra_trace.py](../../../../mindspeed_llm/tools/chakra_trace.py)，支持：

- 读取昇腾 Profiling 生成的`trace_view.json`，提取计算、通信和其他事件。
- 基于显式关联信息（correlation、flow、同一设备流先后顺序）构建保守的执行依赖。
- 导出符合 Chakra schema 的 protobuf 文件，包含`GlobalMetadata`元数据。

转换是离线过程，不修改训练流程，也不需要 NPU 环境。

## 安装依赖

脚本本身只依赖 Python 标准库，不带`--output`参数时可直接运行。导出 protobuf 文件需要安装 Chakra：

```shell
pip install git+https://github.com/mlcommons/chakra.git
```

## 使用方法

在项目根目录执行：

```shell
# 只统计输入 trace 的事件分布，不导出
python mindspeed_llm/tools/chakra_trace.py --input path/to/trace_view.json

# 统计并导出 Chakra protobuf 文件
python mindspeed_llm/tools/chakra_trace.py --input path/to/trace_view.json --output out.chakra
```

| 参数 | 是否必选 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--input` | 是 | 无 | 昇腾 Profiling 生成的`trace_view.json`路径。 |

没有现成 trace 文件时，可用测试目录下的样例[sample_trace_view.json](../../../../tests/ut/tools/sample_trace_view.json)体验完整流程，该文件包含计算、集合通信、点到点通信、correlation 和 flow 事件。
| `--output` | 否 | 无 | Chakra 输出文件路径；不指定时只打印统计信息。 |
| `--format` | 否 | `protobuf` | 输出格式：`protobuf` 为 Chakra 官方二进制格式；`json` 直接生成单个 JSON 文档，字段命名与 `chakra_jsonizer` 一致。 |

## 统计输出字段

脚本向标准输出打印 JSON 统计结果，字段说明如下：

| 字段 | 说明 |
| --- | --- |
| `event_count` | 输入 trace 中的事件总数。 |
| `categories` | 原始事件类别（`cat`字段）的分布。 |
| `phases` | 原始事件阶段（`ph`字段）的分布。 |
| `event_kinds` | 按名称和类别识别出的事件类型分布：`compute`、`communication`、`other`。 |
| `dependency_edge_count` | 构建出的依赖边数量。 |
| `dependency_reasons` | 依赖边的来源分布：`correlation`、`flow`、`same_stream`。 |
| `dependency_warnings` | 依赖分析警告，例如缺少流信息或时间戳的事件数量。 |
| `chakra_node_count` | 实际导出的 Chakra 节点数量，仅在指定`--output`时出现。 |
| `output_path` | 导出文件路径，仅在指定`--output`时出现。 |

## Chakra 节点映射

事件按名称和类别映射为 Chakra 节点类型：

| 事件特征 | Chakra 节点类型 |
| --- | --- |
| 计算事件（kernel、operator 等） | `COMP_NODE` |
| 名称含 send 的通信事件 | `COMM_SEND_NODE` |
| 名称含 recv 或 receive 的通信事件 | `COMM_RECV_NODE` |
| 其他通信事件（HCCL、Hcom 等） | `COMM_COLL_NODE` |
| 其余事件 | `METADATA_NODE` |

集合通信节点会额外写入`comm_type`属性，可识别 allreduce、allgather、reducescatter、alltoall、broadcast。

## 依赖规则与限制

- 依赖只来自输入 trace 中的显式关联：共享 correlation 标识、flow 事件（`ph`为`s`/`t`/`f`）、同一进程同一设备流中的时间先后。
- 不跨设备流根据时间先后臆造依赖。
- 缺少时间戳的事件不参与依赖构建，也不会被导出为 Chakra 节点。
- 缺少进程或流信息的计算、通信事件不参与同流依赖，仅在`dependency_warnings`中汇总计数。

## 结果校验

导出的文件可按 Chakra 官方格式逐条读回校验：每条 message 前有 varint 长度前缀，第一条为`GlobalMetadata`（包含`version`、`schema=Ascend`、起止时间戳），其余为`Node`。已验证的官方消费方式：

```shell
# 转成 JSON 文本查看
chakra_jsonizer --input_filename out.chakra --output_filename out.json

# 生成依赖图（graphml 格式只需 pip 安装 networkx；pdf/dot 还需要系统 graphviz）
chakra_visualizer --input_filename out.chakra --output_filename out.graphml
```

也可以用`chakra.schema.protobuf.et_def_pb2`自行编程解析。依赖图会喂给 ASTRA-sim 等仿真器做 what-if 分析（Chakra 源码 `src/feeder/`中的 C++ ET feeder）。
