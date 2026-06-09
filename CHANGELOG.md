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
- Consolidated GitHub Actions by keeping one CI workflow and one release workflow, with optional PyPI publishing only through the release flow.
- Batch global CLI overrides now fail fast for invalid combinations before any export starts.
- Remote-code export paths now require explicit `--trust-remote-code`; auto-detection warns instead of enabling arbitrary remote code execution.
- Export subpackage imports are lazy so importing `model_exporter.export` no longer eagerly imports the full pipeline stack.
- Wrapper scripts now automatically use the repository `.venv` interpreter when present.
- Added a CI check that keeps `requirements-prod.txt` synchronized with `pyproject.toml` runtime dependencies.
- Updated docs, NOTICE, SECURITY, and MANIFEST metadata to match the current CLI, dependency, logging, and release workflows.
- File logging now honors `FLOUDS_LOG_DIR` and otherwise writes to `./logs/onnx_exports` when explicitly enabled.
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
