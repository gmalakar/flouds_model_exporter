# Production Readiness Audit

Reviewed 2026-11-21 against 1.0.1. The fixes were released as 2.0.0 on 2026-11-24.

**Score: 4 / 10 before, about 7 / 10 after.** The exporter worked on the happy path, but as a
library called by a service it could run remote code without consent, kept callers' tokens,
and could end the calling process.

## Findings and status

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | Critical | `trust_remote_code=False` was raised to `True` on retry | Fixed; raises `RemoteCodeNotAllowedError` |
| 2 | Critical | A per-call token was saved to disk by `huggingface_hub.login` and left in the environment | Fixed; token is an argument for that export only |
| 3 | High | Library code raised `SystemExit` (lock held, validation failed) | Fixed; raises exporter errors |
| 4 | High | Subprocess arguments were type-coerced from strings | Fixed |
| 5 | High | Model reference passed to `git clone` unchecked | Fixed; validated URL after `--` |
| 6 | High | Cleanup deleted other exports' temporary directories | Fixed; only directories older than six hours |
| 7 | High | Fallbacks changed precision silently | Fixed; opt-in and recorded in `export_fallback.json` |
| 8 | High | A failed export gave no reason | Fixed; `ExportFailedError.detail` |
| 9 | High | `from model_exporter import export` was not callable | Fixed |
| 10 | Medium | Exact pins (`onnx`, `psutil`) and `numpy<1.27` | Fixed; ranges verified on both ends |
| 11 | Medium | No test ran a real export; CI had no security scan | Fixed; integration tests and CI jobs added |
| 12 | Medium | No way to pin the model revision | Partly fixed; `revision` option added, see below |
| 13 | High | Errors swallowed: about 350 `except Exception` handlers | Open |
| 14 | Medium | Two pipelines and a legacy fallback coexist; two `helpers.py` | Open |
| 15 | Medium | Unit coverage about 30%; core modules under 20% | Open |
| 16 | Low | `export()` takes over 30 parameters | Open |

## Verified

- 133 unit tests, flake8, black, isort and mypy pass.
- 4 integration tests pass: real export in-process and through the subprocess path, reuse of an
  existing export, and a missing model failing with a reason.
- The same unit and integration tests pass on Python 3.12 with NumPy 2.5.5, onnx 1.23.2,
  onnxruntime 1.32.0 and psutil 7.2.2, and on Python 3.11 with NumPy 1.26.4 and onnx 1.16.1.

## Open items

- **Swallowed errors (13).** The failure reason now reaches the caller, but most handlers still
  turn errors into `None`, `False` or a debug log. Narrowing them needs the missing tests first.
- **Revision pinning (12).** `revision` is passed to the main export. The helper loads used for
  local preparation and for checking `trust_remote_code` still read the default branch, and the
  resolved commit is not recorded with the artifact.
- **Process-wide effects.** `log_to_file` replaces `sys.stdout` and `sys.stderr` and adds a
  handler to the root logger for the duration of the export, and thread-count environment
  variables are set during an in-process export. Concurrent exports in one process share these.
- **Forced downloads.** The in-process export sets `force_download=True`, so the model is
  downloaded again on every export.
- **Command-line token.** `--huggingface_hub_token` is visible in the process list; the
  environment variable is safer.
- **GPU and large models** were not exercised; only a tiny CPU model was exported.
