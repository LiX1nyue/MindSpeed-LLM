# Repository guidance

## Runtime and architecture

- Run launchers and tests from the repository root: scripts resolve `examples/`, configs, and data paths relative to the working directory.
- MCore patches upstream Megatron rather than replacing it wholesale. Import `from mindspeed_llm import megatron_adaptor` before importing Megatron modules (see `pretrain_gpt.py`). Importing `mindspeed_llm` with the default backend applies patches and parses CLI arguments as a side effect.
- Patch implementations live in `mindspeed_llm/core/` and `mindspeed_llm/training/`; feature arguments, validation, and patch registration live in `mindspeed_llm/features_manager/`. Trace the relevant feature and its registration in `mindspeed_llm/features_manager/__init__.py` when changing behavior.
- `MINDSPEED_LLM_VERSION` defaults to `012`. Setting it to `018` redirects the entire `mindspeed_llm` package to `tests/mindspeed_llm/`; that directory is an alternate runtime implementation, not just test fixtures. Confirm the target version before choosing which tree to edit.
- FSDP2 uses `train_fsdp2.py` and `mindspeed_llm/fsdp2/`, with external `fsdp_turbo`. Source `examples/fsdp2/env_config.sh` before launching; it sets `TRAINING_BACKEND=mindspeed_fsdp` to bypass MCore import-time patching. Training accepts a YAML path followed by dotted CLI overrides; CLI values take precedence.

## Environment

- Python must be >=3.10. `requirements.txt` alone does not provision PyTorch, TorchNPU, CANN, MindSpeed, Megatron, or FSDPTurbo. Use `docs/en/pytorch/training/install_guide.md` for setup; use `ci/build_ci_envs.sh` and `.gitcode/workflows/scripts/MindSpeed-LLM_ut_test.sh` for current CI versions (the installation guide shows a release-branch environment).
- The default MCore CI uses Megatron `core_v0.12.1`. External sources are declared as submodules under `3rdparty/`; CI also copies their Python packages into the repository root. Ensure the intended `megatron`, `mindspeed`, and `fsdp_turbo` sources are importable; initializing submodules alone does not expose those packages to Python.
- Source the installed CANN and ATB environment scripts before NPU execution. CI uses `/usr/local/Ascend/ascend-toolkit/set_env.sh` and `/usr/local/Ascend/nnal/atb/set_env.sh`; installation paths can differ.
- `setup.py` attempts to build `ascendspeed_te_ops` using TorchNPU and `ASCEND_TOOLKIT_HOME`, but catches build-setup exceptions. A successful package installation does not prove the native operators were built.

## Focused verification

- Unit tests: `python -m pytest -x --log-cli-level=INFO tests/ut/<area>/test_<name>.py`; append `::TestClass::test_method` or `-k expression` to narrow the run. Full gate UT: `python -m pytest -x --log-cli-level=INFO tests/ut`.
- Even mocked/CPU unit tests under `tests/ut/` load `tests/test_tools/utils.py` through `conftest.py`, which imports Megatron and MindSpeed-LLM. They are not dependency-free tests. `DistributedTest` subclasses launch their own worker processes through the pytest hook; run them with pytest rather than adding `torchrun`.
- Many unit tests load parameter cases from a same-stem `.json` file via `create_testconfig`; inspect that companion file when changing coverage.
- One system test: `python -m pytest 'tests/st/test_st.py::test_st_script[<shell-script-stem>]' -v -x`. Pipeline equivalent: `python -m pytest 'tests/pipeline/st/test_pipeline_st.py::test_st_script[<shell-script-stem>]' -v -x`.
- System tests execute distributed training scripts and compare captured logs with matching JSON baselines in `tests/st/baseline_results/` or `tests/pipeline/st/baseline/`. Inspect the selected script for NPU count and `/data/ci` model/dataset prerequisites before running.
- `tests/test_run.sh` is an environment-specific CI orchestrator: it clones/installs dependencies and copies Megatron before parsing even `--help`, writes under `/home/pipeline_log`, and clears `/data/ci/cache/*`. Use focused pytest commands for local verification.

## Lint and formatting

- Tool configuration is in `pre-commit/pyproject.toml`, not a root `pyproject.toml`. Focused checks: `ruff check --config pre-commit/pyproject.toml <files>` and `ruff format --check --config pre-commit/pyproject.toml <files>`; hooks pin Ruff `0.14.14`.
- `.pre-commit-config.yaml` excludes `tests/mindspeed_llm/` from all hooks and `mindspeed_llm/training/training.py` from Ruff formatting. Preserve these scopes when running direct tool commands.
- Full hooks: `pre-commit run --files <files>`. Ruff hooks auto-fix; the local Gitleaks hook requires a separately supplied `./gitleaks` binary and `pre-commit/.gitleaks.toml`.
