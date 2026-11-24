# Code-R1 Enhanced — 2026-11-24

Baseline: ganler/code-r1, commit `443f8da8058d07051793eff875da117534fd1ba1` (provided Windows clone). Upstream research credits, Apache-2.0 licenses, training implementations and historical reported results are retained. The new toolkit incorporates the previous independently written CodeR1 Lab prototype and extends it with native trainer integration.

## Modified upstream functionality

- `verl/utils/reward_score/coder1`: validates oracle shape before assigning scores; rejects empty/mismatched test arrays and ambiguous completions; checks functional-test completion; avoids logging full code/tests by default; records optional structured reward telemetry. Original reward values are preserved for valid full passes, format-only failures, and invalid formatting.
- Firejail: bounded host capture, explicit network restriction, reduced inherited environment, standardized temp files and timeout handling.
- Docker: no shell interpolation of input/code; read-only mounted task files, non-root container, restricted resources/network and explicit cleanup. Optional image includes NumPy and pytest.
- CES/Kira: explicit request deadlines, bounded retry counts, configurable URLs; failure to reach an executor raises an infrastructure error instead of silently assigning a failed reward.
- PRIME: validates batch dimensions, supports missing `extra_infos`, bounds worker count and propagates Code-R1 evaluator failures to the caller. Naive manager decodes only the response for Code-R1 tasks.
- Dataset preprocessing: normalizes line endings correctly, rejects mismatched tests, fixes TACO split size/metadata, fixes CodeContests selection/undefined argument, and creates output directory.
- GRPO/PPO: environment overrides, input validation, dataset paths, dry-run configuration checks, correct pipeline exit status and quoted Hydra arguments.

## Added functionality

- `coderlab/`: original custom benchmark suite (20 tasks, 66 cases, 20 mutants), deterministic fixture mode, isolated Docker evaluation, OpenAI-compatible model adapters, Pass@k, bootstrap confidence intervals, common-task comparisons, SQLite experiments, REST/CSV and browser console.
- Native training export: JSONL or optional Parquet using the actual upstream `data_source`, `prompt`, `reward_model`, `extra_info` schema. Separate training/validation/held-out test files and SHA-256 manifests. Native dataset audit checks invalid oracles and duplicate normalized prompts.
- Lightweight CI workflow and regression tests independent of GPU stack. Sandbox preflight script and optional Docker image.

## Validation and limitations

See `evidence/TEST_REPORT.md`. No actual GPU model fine-tuning, model inference, live Docker or Firejail execution occurred in the development environment. Changes to those paths are code-reviewed and/or mock-tested, not end-to-end verified. Functional test execution retains upstream shared-process oracle semantics; the completion marker does not prevent a malicious candidate from inspecting or tampering with tests. Use an isolated disposable VM; neither Docker nor Firejail is asserted to be an absolute security boundary. No new model-quality results are claimed.
