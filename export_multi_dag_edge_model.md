# 多 rank DAG 建边模型

本文档定义 `export_multi_dag.cpp` 中计算、集合通信和 P2P 节点的建边语义。边表示执行依赖，不表示单纯的时间先后。

## 1. 节点关系

### 计算节点的出边

- `COMPUTE -> COMPUTE`：同一确认设备 stream 的串行顺序，或 flow、event、task-queue 等显式依赖。
- `COMPUTE -> COLLECTIVE`：计算结果触发集合通信；优先使用 `connection_id`、launch/flow、event 或同 stream 证据。
- `COMPUTE -> P2P`：计算结果被 SEND 使用，或计算完成后通过同步点启动 P2P。

### 集合通信节点的出边

- `COLLECTIVE -> COMPUTE`：通信完成后，计算流通过 event/notify/flow/task queue 等同步点继续执行。
- `COLLECTIVE -> COLLECTIVE`：同一确认通信 stream 的后续通信，或具有明确完成依赖的跨 stream 通信。
- `COLLECTIVE -> P2P`：后续 SEND/RECV 依赖该 collective 的结果或完成同步。

### P2P 节点的出边

- `P2P_RECV -> COMPUTE`：接收完成后消费数据。
- `P2P_RECV -> COLLECTIVE`：接收数据参与后续集合通信。
- `P2P_SEND -> COMPUTE/COLLECTIVE/P2P`：仅在同 stream 或显式同步/任务队列证据成立时建立。

## 2. 证据优先级

建边按以下优先级处理：

1. **显式证据**：`followed_by`、`hostOp`、`HostOpDevice`、`sync_task_queue`、flow、`EVENT_RECORD/EVENT_WAIT`、`Notify_Record/Notify_Wait`。
2. **本卡连接证据**：`connection_id` 只用于同 rank 的 launch 到 HCCL/通信节点链路；`correlation_id` 只用于 enqueue/dequeue。
3. **同设备 stream 顺序**：只在 `Physic Stream Id` 或可靠的设备 stream 上建立；不把普通 display thread 当作串行设备流。
4. **受限推断**：在启用 `--infer-dependencies` 时，使用每个 rank 的 collective 原始记录、完成时间和有限窗口寻找等待/后继节点，并在诊断文件中标记 `inferred_*`。

全局时间相邻、`notify_id` 单独相同、`connection_id` 跨 rank 相同、通信组字符串相同，都不能单独证明依赖。

## 3. 集合通信完成边

跨 rank collective 合并后，节点的时间是所有参与 rank 记录的包络；但后继匹配必须使用每个 `records[]` 的 rank-local `ts` 和 `dur`，不能只使用合并后的 `span_us`。

优先建立：

```text
collective completion evidence -> waiting compute/communication
```

只有在等待事件能映射到公开节点时，才折叠成：

```text
merged_collective -> successor
```

没有显式完成证据时，可以开启：

```bash
--infer-dependencies --dependency-window-us 1000
```

推断边必须保守，并通过 `inferred_collective_completion` 或 `inferred_compute_to_communication` 写入诊断信息。默认不开启推断，避免把通信计算 overlap 错误串行化。

## 4. 边原因

诊断文件中的 `reason` 使用以下名称：

| reason | 含义 |
|---|---|
| `same_stream` | 同一确认设备 stream 的顺序 |
| `flow` | 匹配的 flow 事件 |
| `event_record_wait` | `EVENT_RECORD/EVENT_WAIT` |
| `notify_record_wait` | 带 context、group 和 peer 约束的 Notify |
| `inferred_collective_completion` | collective 完成后的受限推断 |
| `inferred_compute_to_communication` | 计算到通信的受限推断 |

边在最终输出中按目标节点 ID 去重；同一对节点可能有多个证据，但 DAG 文本只保留一条边，诊断文件保留证据记录。

## 5. 不能静默丢弃的关系

- P2P SEND/RECV 不能因为不属于公开 collective 节点就删除；至少应保留节点和 unresolved 诊断。
- Notify 缺少 group、peer 或 context 时不得仅按裸 ID 配对。
- collective 没有出边时，诊断文件会写入 `collective_without_successor`，需要检查是否缺少完成同步证据或是否应启用受限推断。
- 结果必须保持无环；如果新增边导致环，程序失败而不是输出有环图。
