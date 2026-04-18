# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

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
- Introduced CLI subcommand scaffolding (`export`, `validate`, `optimize`, `batch`) with `export` wired and legacy direct-flag invocation preserved.
- Added Python-native `batch` subcommand with memory-aware recommended preset orchestration; `run_exports.ps1` is now optional Windows convenience rather than the primary batch path.
- Made batch orchestration config-driven via `src/model_exporter/config/policy.yaml` and added `--config` support for deterministic YAML-defined export pipelines.
- Converted `run_exports.ps1` into a thin wrapper over the canonical Python `batch` subcommand to eliminate duplicate orchestration logic.
- Implemented `validate` subcommand by forwarding to the shared ONNX validator module instead of keeping it as a stub.
- Implemented `optimize` subcommand by forwarding to the shared optimizer service for existing exported ONNX directories.
- Added `pytest.ini` to scope default discovery to `tests/` and prevent utility scripts under `tools/` from being collected as tests.
- Restricted package discovery to `model_exporter*` so release artifacts do not include repository utility/test modules.
- Added `MANIFEST.in` to keep source distributions focused on runtime package content and exclude non-release directories (tests/tools/examples/logs).
- Constrained runtime support to Python 3.12 only (`requires-python = ">=3.12,<3.13"`) and aligned README/CI/repository settings so unsupported versions fail fast instead of failing during dependency resolution.
- Added a repository-level `.python-version` pin and updated local setup instructions to create `.venv` with `py -3.12`, keeping contributor environments aligned with the supported runtime.
