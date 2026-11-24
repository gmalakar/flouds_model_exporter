# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

## [2.0.1] - 2026-11-24

Documentation only; no code changes.

### Changed

- README: the command-line reference lists all eight `--model-for` types (`vision`, `clip`, `vlm` and `multimodal` were missing), documents `--model-folder` and `--quantize`, and the validate options `--texts`, `--atol`, `--rtol` and `--skip-diagnostics`.
- README: the Python API table lists every `export()` parameter (ten were missing, including `force`, `model_folder`, `library` and `cleanup`).
- `docs/MODEL_EXPORT_COMPLETE_FLOW.md`: the model-type table includes the four newer types.
- The 2.0.0 release date is corrected to 2026-11-24.

## [2.0.0] - 2026-11-24

Breaking changes are marked **Breaking**. See "Upgrading from 1.x" in the README.

### Security

- **Breaking:** `trust_remote_code` is never switched on by the exporter. When a model needed it, the fallback loader and the local snapshot retry used to load again with `trust_remote_code=True` even though the caller had passed `False`, running code from the model repository without consent. They now raise `RemoteCodeNotAllowedError`. The fallback exporter's default for a missing flag also changed from `True` to `False`.
- **Breaking:** a Hugging Face token passed to `export()` is no longer saved. The exporter used to call `huggingface_hub.login`, which writes the token to disk as the machine's default credential, and left `HUGGINGFACE_HUB_TOKEN` set in the process, so one caller's token was used for every later export. The token is now passed to the hub calls as an argument for that export only. In subprocess mode it goes to the child through its environment, not the argument file.
- The model reference is checked before use, and `git clone` receives a validated URL after `--`. A reference starting with `-`, containing control characters, or using a scheme other than `https` for cloning is refused with `InvalidModelReferenceError`.
- Tokens are redacted from one more debug log line that printed the exporter arguments in full.

### Fixed

- **Breaking:** library code no longer ends the caller's process. A second export of the same model used to raise `SystemExit(0)`, reporting success while nothing was exported, and a failed required validation raised `SystemExit(code)`. They now raise `ExportInProgressError` and `ExportValidationError`. The `flouds-export` command keeps its exit codes.
- `from model_exporter import export` returned the `export` sub-package instead of the function once the sub-package was imported, so the documented usage failed with "module object is not callable". It is callable again, and `export_model` is available as an unambiguous name for the same function.
- A failed export now says why: `ExportFailedError` carries the last error reported (`detail`), instead of only "All export attempts failed".
- Subprocess arguments keep their types. String values such as `"123"`, `"none"`, `"true"` and `"1.0"` were converted to numbers, `None` and booleans in the child process, which broke models and folders with such names.
- Temporary directories of other exports are left alone. Cleanup used to remove every matching directory in the system temp folder, or those older than five minutes, which could delete the working directory of an export still running in another process. Only directories older than six hours are removed now.
- The validator no longer passes a missing reference embedding to the diagnostics writer.

### Changed

- **Breaking:** a fallback that changes numeric precision (float16) only runs when `use_fallback_if_failed` is set. Whichever fallback produces the artifact is recorded in `export_fallback.json` next to the model and logged as a warning.
- Dependency ranges widened: `onnx>=1.16.1,<2` (was `==1.16.1`), `numpy>=1.26.4,<3` (was `<1.27`), `psutil>=5.9.8,<8` (was `==5.9.8`). Verified by real exports on NumPy 1.26.4 with onnx 1.16.1 and on NumPy 2.5.5 with onnx 1.23.2.
- The token is passed to `transformers` and `optimum` as `token`, replacing the deprecated `use_auth_token`.

### Added

- `revision` option (`--revision`) to export a specific branch, tag or commit.
- `model_exporter.errors` with `ExporterError` and its subclasses.
- Model types `vision`, `clip`, `vlm` and `multimodal`.
- Integration tests that download a tiny model, export it in-process and through the subprocess path, and run the result (`pytest -m integration tests/integration`).
- CI: a real-export job on Python 3.11 and 3.12, a security job (bandit, pip-audit), and runs on `release/**` branches.

## [1.0.1] and earlier

### Added

- Open-source governance baseline files:
  - LICENSE (Apache-2.0)
  - CONTRIBUTING.md
  - CODE_OF_CONDUCT.md
  - SECURITY.md
  - CHANGELOG.md

### Changed

- Repository legal posture updated for Apache-2.0 readiness.
- Python file headers normalized with SPDX identifiers.
- Requirements naming standardized to `requirements-prod.txt` and `requirements-dev.txt`.
- CLI flags standardized to canonical hyphenated forms only.
- `.gitignore` cleaned to remove over-broad/stale ignore rules.
- CI hardened with workflow concurrency, pip caching, and docs/CLI sanity checks.
- Consolidated GitHub Actions by keeping one CI workflow and one release workflow, with optional PyPI publishing only through the release flow.
- Batch global CLI overrides now fail fast for invalid combinations before any export starts.
- Remote-code export paths now require explicit `--trust-remote-code`; auto-detection warns instead of enabling arbitrary remote code execution.
- Export subpackage imports are lazy so importing `model_exporter.export` no longer eagerly imports the full pipeline stack.
- Wrapper scripts now automatically use the repository `.venv` interpreter when present.
- Added a CI check that keeps `requirements-prod.txt` synchronized with `pyproject.toml` runtime dependencies.
- Updated docs, NOTICE, SECURITY, and MANIFEST metadata to match the current CLI, dependency, logging, and release workflows.
- File logging now honors `LOG_DIR` and otherwise writes to `./logs/onnx_exports` when explicitly enabled.
- Removed stale root release instructions, the obsolete legacy batch preset copy, and the redundant root header helper script.
- Added GitHub issue forms and a pull request template.
- Added maintainer guidance for branch protection and required checks in `docs/GITHUB_REPOSITORY_SETTINGS.md`.
- Added `.github/CODEOWNERS` for review ownership enforcement.
- Added `.github/dependabot.yml` for weekly pip dependency updates.
- Added `.github/workflows/release.yml` for tag-based build and GitHub release automation.
- Added `docs/RELEASE_PROCESS.md` with versioning, tagging, and optional PyPI publish steps.
- Added release guard to enforce tag/version parity between `vX.Y.Z` tags and `pyproject.toml` project version.
- Removed remaining README placeholder clone URL and platform-specific wording for OSS neutrality.
- Hardened `run_exports.ps1` by replacing `Invoke-Expression` with safe argument-array execution and modernized memory query API usage.
- Expanded CI with explicit `type-check` and `packaging-sanity` jobs for pull requests and `main` validation.
- Introduced CLI subcommands (`export`, `validate`, `optimize`, `batch`) and removed legacy direct-flag invocation so every command uses an explicit subcommand.
- Added Python-native `batch` subcommand with memory-aware recommended preset orchestration; `run_exports.ps1` is now optional Windows convenience rather than the primary batch path.
- Made batch orchestration config-driven via `src/model_exporter/config/policy.yaml` and added `--config` support for deterministic YAML-defined export pipelines.
- Converted `run_exports.ps1` into a thin wrapper over the canonical Python `batch` subcommand to eliminate duplicate orchestration logic.
- Implemented `validate` subcommand by forwarding to the shared ONNX validator module instead of keeping it as a stub.
- Implemented `optimize` subcommand by forwarding to the shared optimizer service for existing exported ONNX directories.
- Centralized pytest discovery settings in `pyproject.toml` to keep test configuration in one place.
- Restricted package discovery to `model_exporter*` so release artifacts do not include repository utility/test modules.
- Added `MANIFEST.in` to keep source distributions focused on runtime package content and exclude non-release directories (tests/tools/examples/logs).
- Aligned runtime support to Python 3.11 and 3.12 (`requires-python = ">=3.11,<3.13"`) and updated README/CI guidance accordingly.
- Kept `.python-version` on 3.12 as the preferred local runtime while CI verifies both supported Python versions.
- Release workflow now publishes to PyPI automatically on semantic version tag pushes when PyPI Trusted Publishing is configured.
- Release workflow actions now use Node.js 24-compatible major versions, and manual release reruns validate the supplied tag.
- PyPI publishing now uses the `flouds_by_goutam` GitHub environment to match Trusted Publisher claims.
