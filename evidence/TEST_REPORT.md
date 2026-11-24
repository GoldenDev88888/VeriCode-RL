# Local validation report — Code-R1 Enhanced

Baseline: `443f8da8058d07051793eff875da117534fd1ba1`. Date: 2026-11-24.

## Executed checks

| Check | Actual result |
|---|---|
| `python -m pytest -q enhancement_tests` | 35 passed, 2 skipped (Docker integration) |
| Trusted bundled reference/mutant tests | All 20 reference implementations pass their supplied cases; all 20 deliberate mutants fail at least one case |
| Native JSONL export | 10 train, 4 validation, 6 held-out test tasks |
| Native dataset audit | 20 unique normalized prompts, 0 invalid rows |
| Fixture evaluation | 36 scripted sample outcomes across 2 fixtures and 6 held-out tasks |
| Original adapter regression tests | Mocked Firejail/Docker/remote executor calls, stdin handling, timeouts, invalid-oracle checks, and PRIME failure propagation |
| Subprocess utility | Actual bounded capture tests with trusted constant subprocess commands |
| GRPO/PPO launcher | Both configuration dry runs passed with upstream 8-GPU configuration; invalid GPU count rejected |
| Dashboard | HTTP/API/CSV integration tests using FastAPI TestClient; no browser visual QA |
| Python syntax | 27 modified/new Python modules parsed at final syntax check |
| JavaScript syntax | `node --check coderlab/static/app.js` passed if Node available (see syntax-checks.txt) |

`pytest.txt`, `native-audit.json`, `native-dataset/manifest.json`, `fixture-evaluation.json`, `fixture-audit.json`, `launcher-dry-run.txt` and `demo-experiments.sqlite3` contain recorded outputs. The fixture results are intentionally scripted, not measured model performance. The trusted reference tests execute only authored local fixture solutions; the product's real evaluation path does not execute generated code on the host.

## Not validated in this environment

- No live Docker daemon or Firejail executable was available. Their real isolation and lifecycle were not exercised; two Docker tests were explicitly skipped.
- No GPU training, model checkpoints, external inference API, Hugging Face downloads, Ray distributed execution, or original full test suite was executed.
- PyArrow and the training dependency stack were not installed, so native Parquet export and end-to-end dataset ingestion were not run. The native JSONL schema and row counts were tested instead.
- The original LeetCodeDataset submodule was empty in the supplied archive; no external training data is included.
- No security audit or new benchmark performance gain is claimed. Functional and pytest oracle modes remain susceptible to test interference by adversarial candidate code; use isolated disposable infrastructure.
