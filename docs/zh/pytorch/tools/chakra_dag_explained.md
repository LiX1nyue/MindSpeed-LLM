# 通算 DAG 是怎样生成的：从 trace 解析到依赖图

本文面向没有性能分析经验的读者，预计阅读时间约 **15 分钟**。阅读后，你可以理解：`chakra_trace.py` 这个离线脚本如何把昇腾 Profiling 生成的 `trace_view.json` 一步步变成一张计算+通信依赖图（通算 DAG），以及转换结果写到了哪里。

如果想先了解 trace 文件本身是怎么来的，推荐阅读[Trace 文件生成逻辑通俗说明](trace_generation_explained.md)；命令行参数和字段说明见[Chakra 执行轨迹转换](chakra_trace.md)。

## 1. 什么是通算 DAG

训练一步时，NPU 上既在算矩阵（计算），也在和其他卡交换数据（通信）。把时间线里的事件串成图，就得到通算 DAG：

- **通算** = 通信 + 计算。图里有两种节点：计算节点（MatMul、Add 等算子）和通信节点（allreduce、send/recv 等）。
- **DAG** = 有向无环图。每条边表示“先发生”：A 指向 B，表示按记录 B 排在 A 之后。没有环，因为时间不会倒流。

时间线回答“什么时候发生了什么”，DAG 回答“谁在等谁”。后者是分析计算通信重叠、找关键路径的基础。本工具把昇腾 trace 转成 MLCommons 定义的 Chakra 格式，让 Chakra 生态的分析工具能直接消费这张图。

## 2. 输入和输出一览

| 项目 | 内容 | 位置 |
| --- | --- | --- |
| 输入 | 昇腾 Profiling 生成的 `trace_view.json` | 训练开启 `profile` 后，由采集器写入 `--profile-save-path`（FSDP2 默认 `./profile`）下的产物目录 |
| 输出 | Chakra 文件（protobuf 二进制或 JSON） | 由命令行参数 `--output` 指定，不给则不写文件；`--format json` 可直接生成 JSON |
| 附加输出 | 事件与依赖的统计 JSON | 打印在屏幕上（标准输出） |
| 样例输入 | 可直接体验的样例 trace | `tests/ut/tools/sample_trace_view.json` |

## 3. 六步流水线

转换过程像一条流水线，共六步：

```text
trace_view.json
     ↓  ① 体检：load_trace
原始事件列表
     ↓  ② 分类型：classify_event
     ↓  ③ 填标准卡片：extract_events
统一事件记录
     ↓  ④ 连线：build_dependencies
带依赖边的事件
     ↓  ⑤ 换队服：build_chakra_nodes
Chakra 节点
     ↓  ⑥ 装箱：write_chakra_trace
out.chakra（--output 指定的文件）
```

### ① 体检：读文件并检查结构

先把 JSON 读进来，然后做几项硬性检查：顶层必须是字典、必须有 `traceEvents` 列表、每个事件必须是字典、`cat`/`ph` 必须是字符串。任何一项不合格直接报错退出，不带着疑问往下走。

### ② 分类型：按名字猜工种

昇腾 trace 没有统一的事件类型字段，脚本按“名字 + 类别”的小写文本做匹配：

- 名字或类别含 `hccl`/`hcom` → 通信事件（昇腾集合通信库 HCCL 的痕迹）；
- 含 `kernel`/`operator`/`compute`/`cpu_op`/`npu` → 计算事件；
- 都不沾边 → 其他（例如 `process_name` 这类元数据）。

### ③ 填标准卡片：统一字段

每个事件被抄写成一张标准卡片，字段包括：来源行号、名字、类别、阶段、工种、开始时间（微秒）、时长、进程号、线程号、流号、flow 标识和原始 `args`。缺失时的处理规则：

- 开始时间不是数字 → 记为 `None`（后面不参与连线和导出）；
- 时长不是数字 → 记为 0；
- 流号先从事件顶层 `stream` 找，找不到再去 `args` 里找 `stream`/`stream_id`/`Stream`/`Stream Id`；
- 布尔值一律不当作数字接受（避免 `True` 被当成 1）。

### ④ 连线：只信三种证据

这是生成 DAG 的核心一步。脚本采取**保守原则：没有明确证据就不连线**。它只认三种证据：

| 证据 | 通俗理解 | 规则 |
| --- | --- | --- |
| correlation | 身份证配对 | CPU 算子和 NPU kernel 在 `args` 里带相同编号（`correlation`/`External id`/`Record function id` 等字段之一），说明 kernel 由这个算子发起，连一条边 |
| flow | 接力棒 | 阶段为 `s`/`t`/`f` 且 `id`/`bind_id` 相同的事件是一对异步标记，按时间连线 |
| same_stream | 排队 | 同一进程、同一设备流上的计算/通信事件按开始时间排序，相邻两个连一条边 |

三条限制保证了图的可信度：

- 没有时间戳的事件不参与任何连线（不知道它排在哪）；
- 没有进程或流信息的计算/通信事件不参与“排队”，只在统计里汇总成一条警告；
- **不跨设备流按时间猜依赖**——不同的流是并行车道，trace 里没有证据证明谁先谁后。

连好的边记在目标事件的 `data_dependencies` 里，内容是来源事件的节点号。

### ⑤ 换队服：转成 Chakra 节点

带依赖的事件被转成 Chakra 标准节点，同时做四件事：

1. 丢掉没有时间戳的事件；
2. 剩下的按时间从 1 重新编号，依赖边同步换成新编号；
3. 所有开始时间减去最早时间，归零对齐；
4. 按名字映射节点类型：

| 事件特征 | Chakra 节点类型 |
| --- | --- |
| 计算事件 | `COMP_NODE` |
| 名字含 `send` 的通信事件 | `COMM_SEND_NODE` |
| 名字含 `recv`/`receive` 的通信事件 | `COMM_RECV_NODE` |
| 其他通信事件 | `COMM_COLL_NODE`（集合通信） |
| 其余事件 | `METADATA_NODE` |

集合通信节点还会识别 `comm_type`（allreduce、allgather、reducescatter、alltoall、broadcast）。原事件的来源行号、原类别、流号也一并保留，方便回查。

### ⑥ 装箱：写 protobuf 文件

文件按 Chakra 官方格式写出：第一条是 `GlobalMetadata`（版本号、`schema=Ascend`、起止时间戳），后面逐条是节点；每条 message 前面写一个 varint 长度前缀，和 Chakra 官方工具读写文件的格式一致。父目录不存在时会自动创建。

## 4. 用样例文件走一遍

样例文件 `tests/ut/tools/sample_trace_view.json` 共 13 个事件，三种证据都能演示到：

```text
aten::matmul ──correlation──► MatMul ──排队──► Add ──排队──► LayerNorm   (stream 7)

async_npu(s) ──flow──► async_npu(f)                                      (id=500)

hcom_allReduce                                                          (stream 9，孤立)

HcomSend ──排队──► HcomReceive                                          (stream 10)

UnknownKernel                                                            (无流信息，不连线)
no_time                                                                  (无时间戳，丢弃)
```

实际转换结果：13 个事件 → **10 个节点、5 条边**（correlation 1 条、flow 1 条、同流排队 3 条），外加 2 条警告（1 个事件缺流信息、1 个事件缺时间戳）。可以亲手验证：

```shell
python mindspeed_llm/tools/chakra_trace.py \
    --input tests/ut/tools/sample_trace_view.json \
    --output /tmp/sample.chakra
```

## 5. 输出文件在哪里

- **Chakra 文件**：完全由 `--output` 决定。上面这条命令写到 `/tmp/sample.chakra`；改成 `--output ./out.chakra` 就写到项目根目录。父目录不存在会自动创建。
- **不给 `--output`**：不写任何文件，只在屏幕打印统计 JSON，适合先检查输入质量。
- **统计 JSON**：打印在标准输出，字段含义见[Chakra 执行轨迹转换](chakra_trace.md)的“统计输出字段”一节。
- **输入从哪来**：训练时开启 `profile`，采集器会把 `trace_view.json` 写进 `--profile-save-path`（FSDP2 默认 `./profile`）下的产物目录，把这个文件路径传给 `--input` 即可。

## 6. 已知限制

- 依赖只来自 trace 里的显式证据，不推断跨流关系，也不推断集合通信内部 rank 间的配对关系；
- 事件分类靠名字匹配，命名不典型的算子可能被归入“其他”；
- 本工具完全离线运行，不参与训练，不影响训练性能。

## 7. 源码入口

全部逻辑在 `mindspeed_llm/tools/chakra_trace.py`，步骤与函数的对应关系：

| 步骤 | 函数 | 位置 |
| --- | --- | --- |
| ① 体检 | `load_trace` | 第 13 行 |
| ② 分类型 | `classify_event` | 第 48 行 |
| ③ 标准卡片 | `extract_events` | 第 84 行 |
| ④ 连线 | `build_dependencies` | 第 249 行 |
| ④ 连线辅助 | `add_group_dependencies` / `add_dependency` | 第 211 / 158 行 |
| ⑤ 换队服 | `build_chakra_nodes` | 第 465 行 |
| ⑤ 节点类型 | `get_chakra_node_type` / `get_collective_type` | 第 407 / 435 行 |
| ⑥ 装箱 | `write_chakra_trace` / `encode_varint` | 第 569 / 544 行 |
| 全流程串联 | `export_chakra_trace` | 第 598 行 |
| 命令行入口 | `main` | 第 625 行 |
