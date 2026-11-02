# Chakra 通算 DAG 接入改动说明

> **给服务器上的 AI 的执行说明**
>
> 本文档描述一次已经在一台**无 NPU、无 torch、无 chakra** 的个人电脑上完成的代码改动。
> 代码本身已经写完并通过离线单元测试（46 项），但**从未在真实昇腾环境运行过**。
>
> 你的任务是：
> 1. 把改动同步到服务器仓库（或在服务器上直接应用本文档列出的改动）；
> 2. 按第 7 节的验证清单逐条执行，**如实报告**哪些通过、哪些失败；
> 3. 失败时按第 8 节排查，并回报现象；不要凭猜测修改设计。
>
> 严禁把"离线测试通过"当成"真实环境验证通过"。第 7 节里凡是需要在服务器上
> 才能做的检查，都必须实际执行后再报告。

---

## 1. 这次改动要解决什么问题

**目标**：在训练模型时，生成一张"计算 + 通信"的有向无环图（通算 DAG），并以
MLCommons Chakra 的 execution trace 格式输出。

**关键设计决定**（这一点必须先理解，否则会误判后续所有取舍）：

旧做法是从 `trace_view.json` **反推**依赖关系。这条路有一批无法确定的死结：

- 跨卡集合通信没有全局唯一键（`connection_id`/`correlation` 都是 rank-local）；
- 跨流什么时候可以连边，只能靠 flow/notify 这类弱证据加时间窗兜底；
- 计算与通信到底谁等谁，只能看时间区间重叠。

**这些信息在训练时本来就在进程里，只是没被记录。** 所以本次采用与 Chakra 官方
一致的两段式分工：

```
结构（谁依赖谁）  ← 运行时观测：TorchDispatchMode + torch.distributed 包装 + 融合算子调用点
时长（各跑多久）  ← profiling：trace_view.json 里的 kernel dur / hcom dur
```

两者用**我们自己分配的序号**做 join，而不是靠名字和重叠时间猜。

**融合算子**（`npu_all_gather_base_mm` 这类把通信和计算编译进一个 kernel 的算子）
**不压缩成单节点**，而是展开成「通信节点 + 计算节点」两个**兄弟**（共同前驱、
共同后继），因为它们本来就是并行的。只有当连估算用的带宽都拿不到时，才退化为
单节点，并显式标注 `fused_unexpanded=true`。

---

## 2. 改动清单

### 2.1 新增文件（全部新建）

| 路径 | 行数 | 职责 |
|---|---:|---|
| `mindspeed_llm/tools/chakra_dag/__init__.py` | 35 | 包说明与模块索引 |
| `mindspeed_llm/tools/chakra_dag/ir.py` | 486 | 中间表示：Event / Node / RankGraph / CommMeta / FusedSpec / Timing，含 JSON 序列化 |
| `mindspeed_llm/tools/chakra_dag/fused_ops.py` | 320 | 融合算子展开规则表 + 完整性自检 |
| `mindspeed_llm/tools/chakra_dag/graph.py` | 582 | 建图、跨卡 collective 合并、环检测、传递边归约 |
| `mindspeed_llm/tools/chakra_dag/timing.py` | 519 | 解析 `trace_view.json`，把实测时长贴到节点上，融合体时间拆分 |
| `mindspeed_llm/tools/chakra_dag/writer.py` | 394 | 写 Chakra protobuf（varint 前缀）与合并图 JSON；延迟导入 chakra |
| `mindspeed_llm/tools/chakra_dag/capture.py` | 909 | 运行时采集（**唯一依赖 torch 的模块**，所有 torch 导入都在函数内） |
| `mindspeed_llm/tools/chakra_dag/export.py` | 613 | 编排：定位 trace、建图、写产物、子进程调用 |
| `mindspeed_llm/tools/chakra_dag/hooks.py` | 126 | 训练侧接入点：begin / end / export，含失败降级 |
| `mindspeed_llm/tools/chakra_dag/cli.py` | 253 | 离线命令行 `python -m mindspeed_llm.tools.chakra_dag.cli` |
| `tests/ut/tools/test_chakra_dag.py` | 783 | 46 项离线单元测试 |
| `examples/mcore/qwen3/pretrain_qwen3_8b_tp2_chakra_dag.sh` | 114 | 示例训练脚本（TP=2，8 卡全采） |

### 2.2 修改文件（4 个，共 +138 行）

#### (1) `mindspeed_llm/training/training.py` — MCore 后端接入

**改动 A**：在 `get_profiler()` 函数之前新增 3 个包装函数：

```python
def chakra_dag_begin(args):
    """安装 Chakra DAG 采集钩子。"""
    try:
        from mindspeed_llm.tools.chakra_dag import hooks as chakra_hooks
    except ImportError:
        return None
    recorder = chakra_hooks.begin(args, rank=args.rank)
    if recorder is not None and recorder.degraded:
        print(f"[chakra-dag] rank {args.rank} 采集降级，原因: {recorder.notes()}")
    return recorder


def chakra_dag_end(args):
    """卸下采集钩子并落盘采集记录。"""
    try:
        from mindspeed_llm.tools.chakra_dag import hooks as chakra_hooks
    except ImportError:
        return None
    return chakra_hooks.end(args, rank=args.rank)


def chakra_dag_export(args):
    """起子进程补时长并写 Chakra 产物（仅 rank0）。"""
    try:
        from mindspeed_llm.tools.chakra_dag import hooks as chakra_hooks
    except ImportError:
        return None
    result = chakra_hooks.export(args)
    if result and not result.get("ok", True):
        print(f"[chakra-dag] 导出未成功: {result.get('error') or result.get('stderr_tail', '')}")
    return result
```

**改动 B**：`train()` 里 profiling 启动处（搜索 `if is_profile_enabled():` 后紧跟
`prof = get_profiler()` 的那一处）：

```python
    if is_profile_enabled():
        prof = get_profiler()
        prof.start()

    # --- Chakra 通算 DAG 采集：挂钩子，随 profiling 一起开 ---
    # 必须在 prof.start() 之后、训练循环之前，保证结构记录覆盖整个采集窗口。
    chakra_dag_begin(args)
```

**改动 C**：训练循环结束后（搜索 `if is_profile_enabled(): prof.stop()`，该处目前
在 `while iteration < args.train_iters:` 循环之外）：

```python
    # 采集先落盘，再让 profiler 收尾：
    # 记录不应覆盖 profiler 自己的 FLUSH 与导出阶段。
    chakra_dag_end(args)

    if is_profile_enabled():
        prof.stop()

    # prof.stop() 之后 trace_view.json 才存在，此时才能补时长并写 .et。
    chakra_dag_export(args)
```

> **顺序不要调换**：`end` 必须在 `prof.stop()` 前（否则会把 profiler 的收尾 I/O
> 算子也记进去），`export` 必须在 `prof.stop()` 后（否则 `trace_view.json` 还没生成）。

#### (2) `mindspeed_llm/fsdp2/utils/profiler.py` — FSDP2 后端接入

**改动 A**：import 增加 `from types import SimpleNamespace`。

**改动 B**：`ProfilerConfig` 增加字段（放在 `current_rank: int = 0` 之后）：

```python
    # --- Chakra 通算 DAG 导出 ---
    chakra_dag_export: bool = False
    chakra_dag_output_path: Optional[str] = None
    chakra_dag_format: str = "both"
    chakra_dag_fused_mode: str = "split-estimate"
    chakra_dag_bandwidth_gbps: Optional[float] = None
    chakra_dag_inplace: bool = False
    chakra_dag_timeout_seconds: float = 1800
    chakra_dag_source: str = "auto"
    chakra_dag_ranks: Optional[List[int]] = None
    chakra_dag_max_events: int = 2_000_000
```

并在 `__post_init__` 中补一行：

```python
        if self.chakra_dag_ranks is None:
            self.chakra_dag_ranks = [-1]
```

**改动 C**：`ProfilerManager.__init__` 中，`npu_profiler.profile(...)` 之后追加：

```python
        # --- Chakra 通算 DAG 采集钩子 ---
        self._chakra_hooks = None
        self._chakra_args = None
        if getattr(config, "chakra_dag_export", False):
            try:
                from mindspeed_llm.tools.chakra_dag import hooks as chakra_hooks

                self._chakra_hooks = chakra_hooks
                self._chakra_args = SimpleNamespace(
                    chakra_dag_export=True,
                    world_size=(torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1),
                    rank=config.current_rank,
                    profile_save_path=config.profile_save_path,
                    chakra_dag_max_events=getattr(config, "chakra_dag_max_events", 2_000_000),
                    chakra_dag_output_path=getattr(config, "chakra_dag_output_path", None),
                    chakra_dag_format=getattr(config, "chakra_dag_format", "both"),
                    chakra_dag_fused_mode=getattr(config, "chakra_dag_fused_mode", "split-estimate"),
                    chakra_dag_bandwidth_gbps=getattr(config, "chakra_dag_bandwidth_gbps", None),
                    chakra_dag_inplace=getattr(config, "chakra_dag_inplace", False),
                    chakra_dag_timeout_seconds=getattr(config, "chakra_dag_timeout_seconds", 1800),
                    chakra_dag_source=getattr(config, "chakra_dag_source", "auto"),
                    chakra_dag_ranks=getattr(config, "chakra_dag_ranks", [-1]),
                )
            except ImportError:
                self._chakra_hooks = None
```

**改动 D**：`start()` 末尾（`logger.info_rank0(...)` 之前）：

```python
            if self._chakra_hooks is not None and self._chakra_args is not None:
                self._chakra_hooks.begin(self._chakra_args, rank=self._chakra_args.rank)
```

**改动 E**：`stop()` 改为：

```python
    def stop(self):
        if self.profiler is not None:
            # 采集先落盘，再让 profiler 收尾导出。
            if self._chakra_hooks is not None and self._chakra_args is not None:
                self._chakra_hooks.end(self._chakra_args, rank=self._chakra_args.rank)
            self.profiler.stop()
            # 此时 trace_view.json 才存在，可以补时长并写 .et。
            if self._chakra_hooks is not None and self._chakra_args is not None:
                self._chakra_hooks.export(self._chakra_args)
            logger.info_rank0(f"[RANK {self.config.current_rank}] Profiling stopped. ...")
```

#### (3) `mindspeed_llm/fsdp2/utils/arguments.py` — FSDP2 命令行参数

在 `profile_save_path` 字段之后追加：

```python
    # --- Chakra 通算 DAG 导出 ---
    chakra_dag_export: bool = field(
        default=False,
        metadata={"help": "Export a compute/communication DAG in Chakra format when profiling stops."},
    )
    chakra_dag_output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for Chakra DAG artifacts. Defaults to <profile_save_path>/chakra_dag."},
    )
    chakra_dag_format: str = field(
        default="both",
        metadata={"help": "DAG output format: 'protobuf', 'json', or 'both'. 'json' does not require Chakra."},
    )
    chakra_dag_fused_mode: str = field(
        default="split-estimate",
        metadata={
            "help": "How to split a fused (MC2/CoC) operator's measured duration: "
            "'split-measured', 'split-estimate', or 'merged'."
        },
    )
    chakra_dag_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={"help": "Bandwidth in GB/s used to estimate the comm part of a fused operator."},
    )
    chakra_dag_inplace: bool = field(
        default=False,
        metadata={"help": "Run DAG export in the current process instead of a subprocess (debugging only)."},
    )
    chakra_dag_timeout_seconds: float = field(
        default=1800.0, metadata={"help": "Timeout in seconds for the DAG export subprocess."}
    )
```

#### (4) `mindspeed_llm/fsdp2/train/trainer.py` — FSDP2 传参

`ProfilerConfig(...)` 构造中，`current_rank=current_rank,` 之后追加：

```python
            chakra_dag_export=getattr(args, "chakra_dag_export", False),
            chakra_dag_output_path=getattr(args, "chakra_dag_output_path", None),
            chakra_dag_format=getattr(args, "chakra_dag_format", "both"),
            chakra_dag_fused_mode=getattr(args, "chakra_dag_fused_mode", "split-estimate"),
            chakra_dag_bandwidth_gbps=getattr(args, "chakra_dag_bandwidth_gbps", None),
            chakra_dag_inplace=getattr(args, "chakra_dag_inplace", False),
            chakra_dag_timeout_seconds=getattr(args, "chakra_dag_timeout_seconds", 1800.0),
```

#### (5) `mindspeed_llm/features_manager/functional/profiling.py` — MCore 命令行参数

`ProfilingFeature.register_args` 中，`--profile-save-path` 之后追加：

```python
        # --- Chakra 通算 DAG 导出 ---
        group.add_argument('--chakra-dag-export', action='store_true',
                                         help='export a compute/communication DAG in Chakra format after profiling')
        group.add_argument('--chakra-dag-output-path', type=str, default=None,
                                         help='output directory for DAG artifacts; default <profile-save-path>/chakra_dag')
        group.add_argument('--chakra-dag-format', type=str, default='both',
                                         choices=['protobuf', 'json', 'both'],
                                         help="DAG output format; 'json' does not require the chakra package")
        group.add_argument('--chakra-dag-fused-mode', type=str, default='split-estimate',
                                         choices=['split-measured', 'split-estimate', 'merged'],
                                         help='how to split the measured duration of a fused MC2/CoC operator')
        group.add_argument('--chakra-dag-bandwidth-gbps', type=float, default=None,
                                         help='bandwidth in GB/s used to estimate the comm part of a fused operator')
        group.add_argument('--chakra-dag-inplace', action='store_true',
                                         help='run DAG export in the current process instead of a subprocess (debug)')
        group.add_argument('--chakra-dag-timeout-seconds', type=float, default=1800.0,
                                         help='timeout for the DAG export subprocess')
        group.add_argument('--chakra-dag-max-events', type=int, default=2000000,
                                         help='upper bound on captured events per rank')
```

---

## 3. 依赖安装

`chakra` **不在 PyPI**（`pip install chakra` 只会装到一个 2010 年的 Python2 包）。
正确装法：

```bash
pip install git+https://github.com/mlcommons/chakra.git
```

它会顺带拉 `protobuf`、`networkx`、`pydot`、`graphviz`、`HolisticTraceAnalysis`
和 `grpcio-tools`（用于生成 `et_def_pb2`）。

**如果服务器离线**：在有网的机器上 `pip download`/`pip wheel` 后分发，或采用
方案 C（把 `et_def.proto`、`storage.proto` 收进 `3rdparty/` 自行生成 pb2）。

**只想看 JSON 图可以完全不装 chakra**：`--chakra-dag-format json` 走纯标准库路径，
protobuf 输出会被自动跳过并在报告里记一条 warning。

验证安装是否成功：

```bash
python -c "from chakra.schema.protobuf import et_def_pb2; print(et_def_pb2.ALL_REDUCE, et_def_pb2.COMM_COLL_NODE)"
# 期望输出：0 7
```

---

## 4. 产物说明

训练结束后（以 `--profile-save-path ./profile_dir` 为例）：

```
./profile_dir/chakra_dag/
├── events_rank0.json      # 第 1 阶段产物：运行时采集的结构（人类可读）
├── events_rank1.json
├── rank0.et               # 每 rank 一份 Chakra execution trace（protobuf）
├── rank1.et
├── multi_rank.et          # 合并图，跨卡 collective 已合并为单节点
├── multi_rank_dag.json    # 合并图 JSON（可读、可 diff，保留每条边的来源）
├── manifest.json          # rank↔trace↔device 映射、时间基准、并行组翻译表
└── export_report.json     # 统计、未匹配项、估算占比、错误
```

### 节点类型映射

| 事件 | Chakra `NodeType` | 关键 attr |
|---|---|---|
| AI_CORE / MIX_AIC 等 kernel | `COMP_NODE` | `rank`, `seq`, `time_source` |
| `hcom_allReduce` 等集合通信 | `COMM_COLL_NODE` | `comm_type`, `comm_size`(bytes), `pg_name`, `participants`, `comm_dtype`, `comm_time_us` |
| `hcom_send` / `hcom_receive` | `COMM_SEND_NODE` / `COMM_RECV_NODE` | `comm_src`, `comm_dst`, `comm_tag`, `comm_size` |
| 融合算子展开的通信段 | `COMM_COLL_NODE` | 额外带 `fused_ops`, `fused_order`, `time_split` |

依赖写在 `data_deps`（仿真器读这个），`ctrl_deps` 留空——两者都填会让 feeder 的
依赖计数永不归零。

### `time_source` 的含义（**看结果时务必先看这个**）

| 值 | 含义 |
|---|---|
| `measured` | 直接来自 trace 实测 |
| `estimated` | 由 `bytes / bandwidth` 估算（融合算子的通信段） |
| `unknown` | 没匹配上，未编造数字 |

`export_report.json` 的 `summary.time_sources` 给出各来源计数，`summary.edge_sources`
给出每条边的依据（`tensor` / `sequence` / `collective` / `inferred`）。

---

## 5. 使用方法

### 5.1 MCore（推荐先跑这个）

```bash
# 1) 单卡快速验证（最快跑通链路）
torchrun --nproc_per_node 1 pretrain_gpt.py \
    ... 常规参数 ... \
    --profile --profile-step-start 5 --profile-step-end 6 \
    --profile-ranks 0 --profile-level level1 --profile-with-cpu \
    --profile-save-path ./profile_dir \
    --chakra-dag-export --chakra-dag-format json

# 2) 多卡合并（需要所有 rank）
    --profile-ranks -1 \
    --chakra-dag-export --chakra-dag-format both

# 3) 完整示例见 examples/mcore/qwen3/pretrain_qwen3_8b_tp2_chakra_dag.sh
```

### 5.2 FSDP2

YAML 里加：

```yaml
training:
  profile: true
  profile_step_start: 5
  profile_step_end: 6
  profile_ranks: [-1]
  profile_level: level1
  profile_with_cpu: true
  profile_save_path: ./profile_dir
  chakra_dag_export: true
  chakra_dag_format: both
  chakra_dag_fused_mode: split-estimate
```

### 5.3 离线补跑 / 自检

```bash
# 从已有采集结果重新生成（不重新训练）
python -m mindspeed_llm.tools.chakra_dag.cli \
    --output ./profile_dir/chakra_dag --profile-path ./profile_dir --format both

# 只用 trace 兜底（无运行期采集时；跨卡配对不可靠，报告会标注 trace_only）
python -m mindspeed_llm.tools.chakra_dag.cli \
    --profile-path ./profile_dir --source trace --output /tmp/dag_from_trace

# 读回校验已有 .et
python -m mindspeed_llm.tools.chakra_dag.cli --output ./profile_dir/chakra_dag --check
```

### 5.4 用官方工具消费

```bash
chakra_jsonizer --input_filename ./profile_dir/chakra_dag/multi_rank.et --output_filename /tmp/g.json
chakra_visualizer --input_filename ./profile_dir/chakra_dag/multi_rank.et --output_filename /tmp/g.graphml
```

---

## 6. 设计取舍（需要在评审时知情的部分）

1. **为什么不用官方的 `chakra_converter PyTorch`**
   它的输入是 PyTorch profiler 的 **Execution Trace**（host ET，`1.0.x-chakra.0.0.4`
   schema），torch_npu 不产出；它硬依赖 `et_replay`(PARAM) 和 `HolisticTraceAnalysis`，
   都绑 CUDA；它按 `ncclKernel` / `record_param_comms` 判类型，昇腾的 `hcom_*` 一个都不匹配。
   我们只复用它的 **schema + 文件格式**，建图逻辑自己写。

2. **融合算子为什么展开成兄弟节点而不是串行链**
   MC2 的设计目的就是通信与计算重叠。串行展开会（a）强加一条不存在的依赖，
   （b）让仿真总时长变成 `t_comm + t_compute`，明显大于实测融合耗时。

3. **为什么融合内部时间只能估算**
   `npu_all_gather_base_mm` 在 NPU 上是一个融合 kernel，trace 里只有一段 `dur`。
   三档策略：`split-measured`（trace 里确实拆开了就用实测）→ `split-estimate`
   （`bytes / bandwidth`）→ `merged`（不拆，标 `fused_unexpanded`）。**不假装精确。**

4. **为什么解析 trace 要起子进程**
   上百 MB 的 JSON 会产生大量短命对象，训练进程里这会引发长时间 GC，插到集合通信
   间隙会导致 HCCL 超时。子进程崩了也只影响导出。

5. **为什么 `end` 要在 `prof.stop()` 之前**
   否则采集记录会覆盖 profiler 自己的 FLUSH 与数据导出阶段，混进大量无关 I/O 算子。

6. **时间轴归零**
   昇腾 `ts` 是 Unix epoch 微秒（约 1.7e15），Chakra 的 `start_time_micros` 是 uint64
   且下游按相对时间处理。故归一化到 0 起点，绝对基准存进 `manifest.json`。

### 已知边界（**必须如实告知使用者**）

- **只看得到 Aten 层算子**。被融合进单个 kernel 的计算不会单独出现。
- **融合内部 comm/compute 时序拆分拿不到**，只能测量或估算。
- **通信微步骤按需求忽略**（`Notify_*`、`Reduce_Inline`、`Memcpy` 等不进公共图）。
- **采集有开销**，只适合 1~2 个 step。
- **跨卡合并要求 `--profile-ranks -1`**，且各 rank 的 `events_rank*.json` 能被 rank0 看到
  （共享文件系统）。否则只能得到各 rank 的单卡图。
- **`fused_ops.FUSED_RULES` 需要随 MindSpeed 版本维护**，见第 7.4 节的自检。

---

## 7. 服务器验证清单（**逐条执行并如实报告**）

### 7.0 前置：静态检查（必须全绿）

```bash
cd <repo_root>

# 语法
python -m py_compile mindspeed_llm/tools/chakra_dag/*.py \
    mindspeed_llm/training/training.py \
    mindspeed_llm/fsdp2/utils/profiler.py \
    mindspeed_llm/fsdp2/utils/arguments.py \
    mindspeed_llm/fsdp2/train/trainer.py \
    mindspeed_llm/features_manager/functional/profiling.py

# 格式与静态检查（用仓库钉住的 ruff 版本）
ruff check --config pre-commit/pyproject.toml mindspeed_llm/tools/chakra_dag/ tests/ut/tools/test_chakra_dag.py
ruff format --check --config pre-commit/pyproject.toml mindspeed_llm/tools/chakra_dag/ tests/ut/tools/test_chakra_dag.py
```

预期：`All checks passed!` 与 `N files already formatted`。

### 7.1 离线单元测试

```bash
python -m pytest -x -q tests/ut/tools/test_chakra_dag.py
```

注意：`tests/ut/conftest.py` 会导入 Megatron 与 MindSpeed-LLM，在完整环境里正常。
若在 CPU-only 机器上收集失败，把测试文件复制到独立目录、设置环境变量后单独运行：

```bash
cp tests/ut/tools/test_chakra_dag.py /tmp/ && cd /tmp
CHAKRA_DAG_PKG=<repo>/mindspeed_llm/tools/chakra_dag python -m pytest test_chakra_dag.py -q
```

预期：装了 chakra 时 **46 passed**；没装时 **45 passed, 1 skipped**（skipped 的是
protobuf 往返测试）。

### 7.2 单卡端到端（**最关键**）

```bash
torchrun --nproc_per_node 1 pretrain_gpt.py <常规参数> \
    --profile --profile-step-start 5 --profile-step-end 6 \
    --profile-ranks 0 --profile-level level1 --profile-with-cpu \
    --profile-save-path ./profile_dag_test \
    --chakra-dag-export --chakra-dag-format json
```

检查点（**逐项确认**）：

| # | 检查内容 | 命令 | 期望 |
|---|---|---|---|
| 1 | 训练正常结束，无新增报错 | 看日志 | 与不开 `--chakra-dag-export` 一致 |
| 2 | 采集文件生成 | `ls ./profile_dag_test/chakra_dag/` | 有 `events_rank0.json` |
| 3 | 采集到事件 | `python -c "import json;d=json.load(open('./profile_dag_test/chakra_dag/events_rank0.json'));print(len(d['events']))"` | **远大于 0**（万级） |
| 4 | 有张量依赖 | 同上，统计 `dep_sources` 里 `tensor` 的数量 | **应有大量 `tensor`**。若几乎全是 `sequence`，说明 dispatch 钩子没生效 |
| 5 | 有通信节点 | `--check` 输出里的 `types` | `COMM_COLL_NODE` 计数 > 0（TP/DP>1 时） |
| 6 | 节点有实测时长 | 看 `rank0_dag.json` 的 `timing.source` | 大量 `measured`，少量 `unknown` 可接受 |
| 7 | 图无环 | 看 `export_report.json` 的 `summary.acyclic` | `true` |

**第 4 项是本次设计的核心验证点**：如果 `edge_sources` 里 `tensor` 占比极低，
说明 `TorchDispatchMode` 没真正拦到算子（常见原因：MindSpeed 的 C++ 侧融合算子绕过了
dispatcher），必须回报，不要当作通过。

### 7.3 多卡端到端

```bash
torchrun --nproc_per_node 8 pretrain_gpt.py <常规参数> \
    --profile --profile-step-start 5 --profile-step-end 6 \
    --profile-ranks -1 --profile-level level1 --profile-with-cpu \
    --profile-save-path ./profile_dag_multi \
    --chakra-dag-export --chakra-dag-format both
```

| # | 检查内容 | 期望 |
|---|---|---|
| 1 | 每个 rank 都有 `events_rank<N>.json` | 8 个文件 |
| 2 | 每个 rank 都有 `rank<N>.et` | 8 个文件 |
| 3 | `export_report.json` 的 `merge.collectives_merged` | **> 0**（这是跨卡合并成功的证据） |
| 4 | `merge.collectives_unmergeable` | 尽量为 0；若不为 0，检查 `unmergeable_detail` 的原因 |
| 5 | `merge.cycles` | `0` |
| 6 | `merge.dangling_edges` | `0` |
| 7 | `multi_rank.et` 能被官方工具读回 | `chakra_jsonizer` 成功输出 JSON |
| 8 | 合并节点的 `participants` | 与 `profiler_metadata.json` 的 `parallel_group_info` 一致 |

### 7.4 融合算子覆盖自检（**容易漏，但很重要**）

```bash
python - <<'EOF'
import torch_npu, mindspeed.ops.lcal_functional as lf
from mindspeed_llm.tools.chakra_dag import fused_ops

names = [n for n in dir(torch_npu) if n.startswith("npu_")]
coc = getattr(lf, "coc_ops", None)
if coc is not None:
    names += [n for n in dir(coc) if not n.startswith("_")]
missing = fused_ops.unresolved_comm_candidates(names)
print("未登记但名字暗示有通信的算子：", missing)
EOF
```

期望输出 `[]`。**如果不是空列表**，说明当前 MindSpeed 版本引入了新的融合算子，
需要往 `mindspeed_llm/tools/chakra_dag/fused_ops.py` 的 `FUSED_RULES` 里补规则
（格式见该文件里的现成条目），否则那部分通信会从图里静默丢失。请把缺失列表回报。

### 7.5 FSDP2 路径

按 5.2 的 YAML 跑一次，检查 `./profile_dir/chakra_dag/` 下有产物即可。
FSDP2 的 collective 走 `torch.distributed.fsdp` 内部，本次没有专门验证，
若 `events_rank*.json` 里通信节点为 0，请如实回报——这属于预期外的路径。

---

## 8. 常见问题排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `events_rank*.json` 不存在 | 采集钩子没装上 | 看训练日志里的 `[chakra-dag] ... 采集降级，原因: ...` |
| 日志说"需要 torch" | `capture.py` 的延迟导入失败 | 确认在 torch 已初始化的上下文里调用 |
| `edge_sources` 里几乎全是 `sequence` | dispatch 钩子没拦到算子 | 见 7.2 第 4 项，回报 |
| 通信节点为 0 | `torch.distributed` 包装没生效，或确实无通信 | 检查 TP/DP/PP 配置；回报 |
| `collectives_merged = 0` | 只有 1 个 rank 采集，或 group 名为空 | 用 `--profile-ranks -1`；检查 `parallel_group_info` |
| 大量 `unknown` 时长 | trace 与采集的算子名对不上 | 正常现象的一部分；若超过 50% 请回报 |
| `chakra 不可用，已跳过 protobuf` | 没装 chakra | 按第 3 节安装，或用 `--chakra-dag-format json` |
| 导出超时 | trace 太大 | 调大 `--chakra-dag-timeout-seconds`，或缩小采集窗口 |
| 训练变慢很多 | 采集开销 | 正常；只采 1~2 个 step，或减小 `--chakra-dag-max-events` |

**导出失败绝不能中断训练**——这是硬约束。若发现导出失败导致训练退出，属于 bug，请回报。

---

## 9. 已在本机完成 / 未完成的事

**已完成（在这台无 torch / 无 NPU / 无 chakra 的个人电脑上）**：

- 全部代码编写，`py_compile` 通过；
- `ruff check` / `ruff format --check`（仓库钉住的 0.14.14）全绿；
- 46 项离线单元测试通过，覆盖：IR 序列化、融合展开（含兄弟拓扑断言）、
  建图、环检测、传递边归约、多卡合并、时间贴附、trace 解析过滤、CLI 错误路径；
- 用**本地生成的最小 chakra pb2**（从官方 `et_def.proto` 编译）验证了
  `.et` 写出→读回→依赖完整性，`GlobalMetadata.version = 1.0.0`；
- 用合成数据跑通了 `begin → end → export` 完整生命周期；
- 用合成数据跑通了 2 rank 合并、融合拆分（comm 41.94us + comp 858.06us 落在
  40~940us 包络内）、`--check` 读回。

**未完成（必须在服务器上做）**：

- 真实 `trace_view.json` 的字段校验（`Task Type`、`Process Name`、
  `hcom_<type>__<comm_name>_<seq>` 命名格式）；
- `TorchDispatchMode` 在真实 Megatron/MindSpeed 训练下是否真能拦到算子；
- `get_hccl_comm_name()` 返回值与 trace 里 `comm_name` 是否真能对上（**这是 join 的
  关键假设，若对不上，时长会大量落到 `unknown`**）；
- 融合算子在真实模型下是否都命中 `FUSED_RULES`；
- 真实多卡跨卡合并的成功率；
- 采集开销对训练的实际影响。

**以上任一项与设计假设不符时，请回报现象而不是自行改设计。**

---

## 10. 后续可做的事（不在本次范围）

1. AGENTS.md 与 `docs/zh/pytorch/tools/` 下 `chakra_trace.py` 的引用已过期
   （该文件在 46bb6d8 被删除），建议后续一并更新为指向 `chakra_dag` 包。
2. 若需要在不使用运行时采集的情况下按模型配置直接生成 DAG，那是 `torch.fx` /
   模型 IR 路线，与本方案正交，可另立需求。
3. `fused_ops.FUSED_RULES` 建议接入 CI 自检（7.4 那段脚本可直接改成用例）。
