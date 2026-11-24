# Code-R1 Enhanced — developer guide

This is the supplied Code-R1 repository with targeted modifications, not a replacement for the original verl trainer. Keep upstream attribution and license files when redistributing. The local benchmark tooling is independent of CUDA. Upstream training is a separate environment and should be configured on Linux with compatible GPUs.

## 1. Windows or Linux: local console

Use Python 3.10+. Extract the archive and open its root in a terminal.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lab.txt
python -m pytest -q enhancement_tests
python -m coderlab.cli evaluate --fixture
python -m coderlab.cli serve
```

Linux/macOS: activate with `source .venv/bin/activate` instead. Open `http://127.0.0.1:8000`; `/docs` is the API explorer. The server binds only to loopback. It has no authentication; do not expose it publicly. Fixture evaluation is deterministic and does not execute model-generated code. It is for pipeline and UI checks, not evidence of AI model accuracy.

## 2. Real inference and evaluation

Run a compatible inference server (e.g. vLLM) on an approved machine. Set its URL and model identifier. Install Docker and pull the runner image before evaluating. Use a disposable isolated VM with no sensitive files/credentials. The runner uses no network and expects numeric/string/list/dict JSON-compatible return values. Expected outputs are compared outside the container. Candidate code can still be adversarial; Docker is not a complete security boundary.

```bash
docker pull python:3.11-slim
python -m coderlab.cli audit
python -m coderlab.cli evaluate --model your-model-id --base-url http://localhost:8001/v1 --split test --samples 5 -k 1
python -m coderlab.cli serve
```

Set `CODERLAB_API_KEY` in your environment if the server requires authentication; never commit credentials. Multiple `--model` values run multiple models. Use the exact same dataset and sampling settings for comparisons. The dashboard reports individual valid-task results and common-task comparisons; infrastructure errors exclude tasks, not count as incorrect outputs. Inspect coverage before interpreting scores. Bootstrap intervals describe sample variation over tasks; small toy datasets are not generalizable performance estimates.

## 3. Native Code-R1 data integration

The JSON task dataset is organized into train (10), validation (4), and held-out test (6). It is deliberately small for development, not sufficient for meaningful large model training.

```bash
python -m coderlab.cli export-verl --output data/coderlab-custom
python -m coderlab.cli audit-native data/coderlab-custom/train.jsonl data/coderlab-custom/validation.jsonl data/coderlab-custom/test.jsonl
python -m pip install 'pyarrow>=15'
python -m coderlab.cli export-verl --parquet --output data/coderlab-custom
python -m coderlab.cli audit-native data/coderlab-custom/train.parquet data/coderlab-custom/validation.parquet data/coderlab-custom/test.parquet
```

The original `verl` RL dataset reader accepts Parquet. Exported rows preserve its native schema, including functional test oracle JSON. Avoid using the held-out test split as a training or tuning split. The audit detects exact normalized prompt duplication, not semantic contamination. The previous upstream corpus uses LeetCode and TACO; its source datasets and checkpoints are **not bundled**. The uploaded submodule directory was empty. To restore upstream source data in a real Git clone:

```bash
git clone https://github.com/ganler/code-r1.git
cd code-r1
git checkout 443f8da8058d07051793eff875da117534fd1ba1
git submodule update --init --recursive
```

Apply the accompanying enhancement patch in that clone or copy enhanced files there. Follow upstream dataset preparation instructions (`python examples/data_preprocess/coder1.py`) after installing its dependencies and verifying dataset access/licensing. Use `audit-native` before training; it is schema/duplicate validation, not a proof that every reference solution passes. Use a separate reference/mutation audit for execution evidence.

## 4. Original GRPO/PPO training

Follow the upstream environment setup in README.md and the verl documentation. The historical README suggests vLLM 0.7.3 while package metadata pins vLLM <=0.6.3; the upstream dependency conflict has not been resolved or GPU-tested here. Choose a tested compatible environment and review `docs/README_vllm0.7.md`. Do not assume arbitrary modern versions are compatible. GPU weights, datasets, and API credentials are not included.

First use a sandbox preflight after installing the training dependencies:

```bash
# Firejail: use upstream install instructions
export CODER1_EXEC=firejail
python -m scripts.check_coder1_executor

# Or use the enhanced Docker adapter
# docker build -f docker/Dockerfile.coder1-sandbox -t coder1-sandbox:local .
# export CODER1_EXEC=docker CODER1_DOCKER_IMAGE=coder1-sandbox:local
# python -m scripts.check_coder1_executor
```

Configure training. Default upstream settings were tuned for 8 H200 GPUs. The following dry run only validates arithmetic and paths; it does not allocate GPUs or confirm memory fit.

```bash
GPUS_PER_NODE=8 CODER1_DRY_RUN=1 bash main_grpo.sh
GPUS_PER_NODE=8 CODER1_DRY_RUN=1 bash main_ppo.sh
```

For actual training on a correctly provisioned GPU system, use a sufficiently sized validated native corpus, the corresponding validation split, and a compatible local checkpoint. Adjust rollout, response length, microbatch and gradient accumulation for actual VRAM. Do not use the toy ten-task corpus with default batch size 16.

```bash
mkdir -p runs
export CODER1_REWARD_LOG="$PWD/runs/rewards.jsonl"
export CODER1_REWARD_WORKERS=16 CODER1_CASE_WORKERS=2
export DATASET=code-r1-12k
export MODEL_PATH=./models/Qwen2.5-7B-Instruct-1M
# Override if using separate validation rather than the historical upstream test file:
# export TRAIN_FILE=/path/to/train.parquet VAL_FILE=/path/to/validation.parquet
bash main_grpo.sh
# or: bash main_ppo.sh
```

Other configurable settings: `GPUS_PER_NODE`, `MAX_EPOCHS`, `ROLLOUT_N_SAMPLE`, `ROLLOUT_N_QUERY`, `MICRO_BATCH_PER_GPU`, `GRAD_ACC_STEPS`, `CODER1_DOCKER_IMAGE`, `CODER1_CES_URL`, `CODER1_KIRA_URL`. Extra arguments pass through to Hydra. A reward infrastructure error now fails the code batch; inspect the error and resume only after restoring infrastructure. The backend timeout may still conflate startup delay and actual candidate timeout; monitor under load. Remote CES/Kira responses cannot prove their isolation/security.

## 5. Security and results interpretation

The original functional and pytest oracle modes share an execution process or filesystem with the candidate. The new marker catches an early exit before tests finish, but malicious candidates can potentially interfere with the test code. The custom Docker grader has host-side expected-output comparisons, yet still cannot prevent every form of malicious resource use. Use dedicated isolated machines/VMs; never run generated code on your workstation directly. No network-enabled or plain-host fallback is provided in the custom grader.

This release contains local regression evidence, not new published benchmark gains. Original model results are attributed to upstream. Only claim personal contributions you actually made, tested, or reproduced; cite the original authors and label this as an enhanced derivative.
