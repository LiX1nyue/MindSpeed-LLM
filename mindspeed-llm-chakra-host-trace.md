# MindSpeed-LLM Chakra Host Trace 改动交接文档

## 目的

本补丁为容器 `8b19e009b00a` 中的 MindSpeed-LLM 增加可选的 PyTorch
`ExecutionTraceObserver` 采集能力。它只负责生成 Chakra 的 **host execution
trace**，不修改 CANN/NPU profiler，也不声称已经完成 NPU device trace 适配。

当前环境没有真实 NPU，容器也没有挂载到本仓库，因此本补丁没有进行训练运行验证。
服务器上的执行者必须在应用补丁后完成真实 NPU 小模型验证。

## 补丁文件

补丁位于：

```text
patches/mindspeed-llm-chakra-host-trace.patch
```

它应当在 MindSpeed-LLM 根目录 `/workspace/MindSpeed-LLM` 中应用：

```bash
cd /workspace/MindSpeed-LLM
git apply /path/to/mindspeed-llm-chakra-host-trace.patch
```

补丁目标版本是本次检查到的容器源码布局，重点文件为：

```text
mindspeed_llm/training/arguments.py
mindspeed_llm/training/training.py
mindspeed_llm/training/chakra_host_trace.py  # 新增
```

## 精确改动内容

### 1. 新增 `ChakraHostTraceManager`

新增文件 `mindspeed_llm/training/chakra_host_trace.py`，功能如下：

- 每个进程创建一个 `torch.profiler.ExecutionTraceObserver`；
- 根据分布式 rank 写入独立文件 `host.<rank>.json`；
- 默认输出目录为 `./chakra_traces`；
- 支持 global step 起止窗口；
- 支持只采集指定 rank；
- 不覆盖已有输出文件；
- 用 `atexit` 做退出兜底清理；
- 正常停止时调用 `stop()` 和 `unregister_callback()`。

注意：observer 记录的是 PyTorch execution trace，不是 CANN 的 `trace_view.json`
或 profiler DB。

### 2. 新增命令行参数

在 `mindspeed_llm/training/arguments.py` 的 `process_args_v2()` 中增加：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--chakra-host-trace` | 关闭 | 开启 host trace |
| `--chakra-host-trace-dir` | `./chakra_traces` | 输出目录 |
| `--chakra-host-trace-start-step` | `0` | 起始 global step，包含该步 |
| `--chakra-host-trace-end-step` | `-1` | 结束 global step，不包含；`-1` 表示训练结束 |
| `--chakra-host-trace-ranks` | `-1` | 要采集的 rank；`-1` 表示所有 rank |

### 3. 修改主训练路径

在 `mindspeed_llm/training/training.py`：

1. 导入 `ChakraHostTraceManager`；
2. `train()` 中创建 manager；
3. 每轮 `train_step()` 前调用 `start_if_needed(iteration)`；
4. 每轮完成、`iteration` 已递增后调用 `stop_if_needed(iteration)`；
5. 训练循环结束时调用 `stop()`。

对于 `start=5,end=6`，行为是：

- 在 global step 5 的 `train_step()` 前启动；
- 完成该步后 iteration 变成 6；
- 调用 `stop_if_needed(6)` 停止；
- 输出只覆盖一个训练 step。

## 运行示例

在真实 NPU 服务器上，先用一个小模型和短训练区间验证：

```bash
torchrun ... pretrain_gpt.py \\
  ... \\
  --chakra-host-trace \\
  --chakra-host-trace-dir /output/chakra_traces \\
  --chakra-host-trace-start-step 5 \\
  --chakra-host-trace-end-step 6 \\
  --chakra-host-trace-ranks 0
```

预期生成：

```text
/output/chakra_traces/host.0.json
```

如果要为所有 rank 采集，使用：

```bash
--chakra-host-trace-ranks -1
```

不要在同一输出目录中重复运行而不清理旧的 `host.<rank>.json`，因为 manager
会主动拒绝覆盖文件。

## 应用补丁后的验证清单

### 参数和导入检查

```bash
python -m py_compile \\
  mindspeed_llm/training/chakra_host_trace.py \\
  mindspeed_llm/training/arguments.py \\
  mindspeed_llm/training/training.py
```

用训练脚本的 `--help` 确认能看到五个 `--chakra-host-trace*` 参数。

### 真实训练验证

服务器执行者必须确认：

1. `host.<rank>.json` 文件被生成且大小大于 0；
2. JSON 可以被 Python `json.load()` 读取；
3. 顶层至少包含 `schema` 和 `nodes`；
4. `nodes` 非空，存在根节点和 `ctrl_deps`；
5. 多 rank 时每个目标 rank 都生成文件；
6. 输出文件的 step 范围符合起止参数；
7. 同一进程已有输出时，程序会明确报 `FileExistsError`；
8. 启用 CANN profiler 时，CANN 输出仍然正常生成。

### Chakra 兼容性验证

在 Chakra 环境中执行：

```python
from chakra.src.trace_link.chakra_host_trace_loader import ChakraHostTraceLoader

ops = ChakraHostTraceLoader().load("/output/chakra_traces/host.0.json")
print(len(ops))
```

然后检查 schema 是否属于 Chakra converter 支持的列表。若 PyTorch 2.7.1
产生了新的 schema，需要在 Chakra 的 `src/converter/pytorch_node.py` 中增加
对应解析版本；这不是本补丁自动修改的内容。

## 已知范围和未改动部分

- 本补丁只改 MindSpeed-LLM 的 `mindspeed_llm/training` 主训练路径。
- 没有修改 `mindspeed_llm/fsdp2` 的 `Trainer`。
- 没有修改旧的 `megatron/training/training.py` PyTorch profiler 路径。
- 没有修改 `torch_npu.profiler` 或 CANN。
- 没有生成 Chakra device trace。
- 没有把 HCCL 事件转换为 Chakra 的 `COMM_COLL_NODE`、`COMM_SEND_NODE` 或
  `COMM_RECV_NODE`。
- 没有保证 host trace 与 Ascend device trace 已经可以被当前 Chakra
  `trace_link` 直接合并。

## 服务器执行者的注意事项

1. 先保存 MindSpeed-LLM 当前工作树：

   ```bash
   git status --short
   git diff > /tmp/mindspeed-before-chakra.patch
   ```

2. 如果源码已被其他改动，先人工解决 patch 冲突，不要覆盖现有工作。
3. 先用单 rank、小模型、单 step 验证 host trace，再扩大到多 rank。
4. 如果训练从 checkpoint 恢复，起止参数对应的是恢复后的 global iteration。
5. host trace 生成成功后，再单独检查 CANN trace 的字段和 HCCL 事件，不能把
   CANN profiler 目录直接重命名成 Chakra 输入文件。
