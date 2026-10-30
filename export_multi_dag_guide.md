# `export_multi_dag.cpp` 使用说明

## 1. 用途

`export_multi_dag.cpp` 是一个**无第三方依赖的 C++ 离线工具**，用于将 8 个 rank 的 Ascend profiler trace 合并为一个计算/通信 DAG。

工具会：

- 从每个 rank 的 `view_trace.json` 中提取计算、集合通信和 P2P 事件；
- 根据 `profiler_metadata.json` 校验 rank 和 `world_size`；
- 在同一设备 stream 上建立有时间先后的依赖；
- 使用 trace 中的 flow、`EVENT_RECORD/EVENT_WAIT` 建立可确认的依赖；
- 将各 rank 中同名的集合通信事件合并为一个 collective 节点；
- 输出文本格式 DAG，并检查结果是否存在环。

该工具适合在服务器完成 profiling 后，将结果复制到离线分析环境，或直接在服务器上运行。它不需要 Python、PyTorch、TorchNPU、CANN、Megatron、MindSpeed 或 FSDPTurbo。

## 2. 输入目录

程序固定读取 8 个目录：

```text
<profile_bundle>/
├── rank0/
│   ├── view_trace.json
│   └── profiler_metadata.json
├── rank1/
│   ├── view_trace.json
│   └── profiler_metadata.json
...
└── rank7/
    ├── view_trace.json
    └── profiler_metadata.json
```

其中：

- `view_trace.json` 可以是顶层事件数组，也可以是包含 `traceEvents` 数组的对象；
- `profiler_metadata.json` 必须包含：

  ```json
  {
    "distributed_args": {
      "rank": 0,
      "world_size": 8
    }
  }
  ```

- `rankN` 目录名中的 `N` 必须与 metadata 中的 `distributed_args.rank` 一致；
- 当前程序只支持 `world_size=8`，并且要求 `rank0` 到 `rank7` 全部存在；
- 集合通信参与 rank 信息从 metadata 中的 `parallel_group_info` 或 `group_info` 读取，支持其中的 `global_ranks` 或 `group_global_rank` 字段。

建议在服务器上先检查目录和文件是否齐全：

```bash
PROFILE_BUNDLE=/path/to/profile_bundle

for rank in $(seq 0 7); do
    test -s "$PROFILE_BUNDLE/rank${rank}/view_trace.json" || \
        echo "missing view_trace.json for rank ${rank}"
    test -s "$PROFILE_BUNDLE/rank${rank}/profiler_metadata.json" || \
        echo "missing profiler_metadata.json for rank ${rank}"
done
```

不要直接打印完整 trace；生产环境中的 trace 可能很大并包含敏感信息。需要检查字段时，应使用有界的 `grep`、脚本或抽样工具。

## 3. 编译

在项目根目录执行：

```bash
cd /path/to/MindSpeed-LLM
mkdir -p /tmp/opencode
g++ -O2 -Wall -Wextra export_multi_dag.cpp -o /tmp/opencode/export_multi_dag
```

如需将可执行文件保存在当前目录，也可以使用：

```bash
g++ -O2 -Wall -Wextra export_multi_dag.cpp -o export_multi_dag
```

编译器只需支持 C++11 或更新标准；源码使用自带 JSON 解析器，不需要链接额外库。

## 4. 运行

命令格式：

```bash
<exporter> <profile_bundle> <dag_output>
```

示例：

```bash
/tmp/opencode/export_multi_dag \
    /path/to/profile_bundle \
    /path/to/output/multi_rank_dag.txt
```

程序会同时生成两个文件：

```text
/path/to/output/multi_rank_dag.txt
/path/to/output/multi_rank_dag.txt.diagnostics.txt
```

成功时标准输出类似：

```text
Successfully exported multi-rank DAG to /path/to/output/multi_rank_dag.txt (Nodes: N, Edges: M)
```

返回码：

- `0`：成功；
- `1`：输入、元数据、匹配或 DAG 校验失败；
- `2`：命令行参数错误。

## 5. 输出格式

输出第一行是：

```text
N M
```

其中 `N` 是节点数，`M` 是边数。之后每行一个节点，`children` 保存该节点的后继节点 ID。

### 计算节点

```text
id=... type=COMPUTE rank=... name="..." task_type=... stream=... ts=... compute_time_us=... out_degree=... children=[...]
```

主要字段：

- `rank`：节点所在 rank；
- `task_type`：例如 `AI_CORE`、`AI_VECTOR_CORE`、`MIX_AIC`、`MIX_AIV`；
- `ts` 和 `compute_time_us`：单位为微秒。

### P2P 节点

```text
id=... type=P2P rank=... name="..." p2p_type=SEND|RECV stream=... ts=... dur=... out_degree=... children=[...]
```

P2P 节点保留在输出中，但当前版本不会自动匹配不同 rank 之间的 send/recv 对。

### 集合通信节点

```text
id=... type=COLLECTIVE name="..." comm_type=ALL_REDUCE participants=[...] ts=... span_us=... participant_source=exact_name records=[...] out_degree=... children=[...]
```

集合通信节点包含：

- `comm_type`：`ALL_REDUCE`、`ALL_GATHER`、`REDUCE_SCATTER`、`ALL_TO_ALL` 或 `BROADCAST`；
- `participants`：根据 group metadata 验证后的 rank 列表；
- `span_us`：参与 rank 中最早开始到最晚结束的时间跨度；
- `records`：每个参与 rank 的原始时间、stream、进程/线程和字节数信息；
- `estimated_bandwidth_GBps`：基于 `bytes / dur_us * 1e-3` 的估算值，单位是 GB/s，不是 Gbit/s。

## 6. 诊断文件

`<dag_output>.diagnostics.txt` 用于审计匹配和依赖恢复过程。重点查看以下信息：

- `unresolved_group`：集合通信找不到可验证的 group 成员；
- `unresolved_p2p`：发现 P2P，但没有进行跨 rank peer 匹配；
- `unresolved_notify`、`unresolved_event_sync`：同步事件缺少足够的上下文；
- `flow missing ...`：flow 事件不完整或无法映射到公开节点；
- `edge ... reason=...`：输出边及其来源，例如 `same_stream`、`flow`、`event_record_wait`。

出现诊断信息不一定代表程序失败，但说明 DAG 可能缺少部分依赖。若程序最终报错，应先查看诊断文件，再检查输入 trace 和 metadata。

## 7. 当前实现的重要限制

1. **集合通信通过完整事件名匹配。** 当前实现将同名 collective 视为同一个跨 rank 操作，即 `participant_source=exact_name`。如果不同 rank 的事件名带有 rank、序号或其他不一致后缀，可能无法合并。
2. **不是通用的跨 rank collective matcher。** 程序不会仅凭时间接近、`connection_id`、local sequence 或 correlation 自动匹配集合通信。
3. **只建立有证据的依赖。** 不会因为两个事件时间相邻就建立跨 stream 或跨 rank 边，也不会把所有事件按全局时间串行化。
4. **P2P 仅保留节点。** 当前版本会报告未解析的 P2P 依赖，不会根据 send/recv 名称自动构造跨 rank 边。
5. **当前固定为 8 rank。** 如果要支持其他 world size，需要修改源码中的 rank 循环、metadata 校验和 group 成员范围检查。
6. **通信字节数可能是估算值。** 当 trace 只有 `count` 和 `data_type` 时，程序使用 dtype 大小估算 bytes；未知 dtype 时不会猜测字节数。
7. **`communication.json` 未参与计算。** 输出中的通信时间来自 trace 事件；它不等同于真实网络 transit time，也不替代通信库提供的带宽数据。

因此，当前输出应作为**保守的、可审计的 DAG 基线**。在服务器上拿到新的 profiler schema 后，应先抽样确认字段，再决定是否扩展匹配规则。

## 8. 最小操作流程

```bash
# 1. 进入源码目录
cd /path/to/MindSpeed-LLM

# 2. 编译
mkdir -p /tmp/opencode
g++ -O2 -Wall -Wextra export_multi_dag.cpp \
    -o /tmp/opencode/export_multi_dag

# 3. 准备并检查 8-rank profile bundle
PROFILE_BUNDLE=/path/to/profile_bundle
for rank in $(seq 0 7); do
    test -s "$PROFILE_BUNDLE/rank${rank}/view_trace.json" || exit 1
    test -s "$PROFILE_BUNDLE/rank${rank}/profiler_metadata.json" || exit 1
done

# 4. 导出 DAG
mkdir -p /path/to/output
/tmp/opencode/export_multi_dag \
    "$PROFILE_BUNDLE" \
    /path/to/output/multi_rank_dag.txt

# 5. 查看摘要和诊断信息
head -n 5 /path/to/output/multi_rank_dag.txt
less /path/to/output/multi_rank_dag.txt.diagnostics.txt
```
