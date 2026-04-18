# Wrapper Scripts Guide

Use this guide to run batch exports through the convenience wrappers on Windows, Linux, and macOS.

## Purpose

The wrapper scripts forward to the canonical Python batch CLI and provide:

- Convenience flags for common export overrides
- Optional config file selection
- Optional text-file import into a temporary batch preset
- Consistent invocation on different operating systems

Canonical CLI command shape:

```bash
python -m model_exporter.cli.main batch --preset <name> [options]
```

## Windows Wrapper

Script: `run_exports.ps1`

### Parameters

- `-UseVenv`: use `./.venv/Scripts/python.exe`
- `-Force`: apply `--force` to each batch export
- `-Optimize`: apply `--optimize`
- `-Cleanup`: apply `--cleanup`
- `-SkipValidator`: apply `--skip-validator`
- `-PruneCanonical`: apply `--prune-canonical`
- `-Portable`: apply `--portable`
- `-NoLocalPrep`: apply `--no-local-prep`
- `-FailFast`: apply `--fail-fast` (stop on first failure)
- `-Config <path>`: use a YAML/JSON batch config file
- `-TextFile <path>`: parse text commands and build a temporary batch config
- `-Preset <name>`: preset name in the selected config (default: `recommended`)
- `-MinFreeMemoryGB <int>`: minimum free RAM threshold for batch (default: `1`)
- `-LogToFile`: request per-export log files and print the logfile path (PowerShell wrapper only)

### Examples

```powershell
powershell -ExecutionPolicy Bypass -File .\run_exports.ps1 -UseVenv -FailFast
```

```powershell
powershell -ExecutionPolicy Bypass -File .\run_exports.ps1 -Config .\docs\batch_presets_example.yaml -Preset text-import -Force -Optimize -Cleanup -FailFast
```

```powershell
powershell -ExecutionPolicy Bypass -File .\run_exports.ps1 -TextFile .\docs\batch_commands.txt -Preset text-import -FailFast
```

## Linux/macOS Wrapper

Script: `run_exports.sh`

### Parameters

- `--use-venv`: use `./.venv/bin/python`
- `--force`: apply `--force` to each batch export
- `--optimize`: apply `--optimize`
- `--cleanup`: apply `--cleanup`
- `--skip-validator`: apply `--skip-validator`
- `--prune-canonical`: apply `--prune-canonical`
- `--portable`: apply `--portable`
- `--no-local-prep`: apply `--no-local-prep`
- `--fail-fast`: apply `--fail-fast` (stop on first failure)
- `--config <path>`: use a YAML/JSON batch config file
- `--text-file <path>`: parse text commands and build a temporary batch config
- `--preset <name>`: preset name in the selected config (default: `recommended`)
- `--min-free-memory-gb <int>`: minimum free RAM threshold for batch (default: `1`)
- `--log-to-file`: request per-export log files and print the logfile path (shell wrapper only)

### Examples

```bash
chmod +x ./run_exports.sh
./run_exports.sh --use-venv --fail-fast
```

```bash
./run_exports.sh --config ./docs/batch_presets_example.yaml --preset text-import --force --optimize --cleanup --fail-fast
```

```bash
./run_exports.sh --text-file ./docs/batch_commands.txt --preset text-import --fail-fast
```

## Text File Mode

Both wrappers support text-file import mode.

Expected format:

- One export command per line
- New hyphenated parameter style only
- Comment lines that start with `#` are ignored

Example line:

```text
flouds-export export --model-name BAAI/bge-base-en-v1.5 --model-for fe --task feature-extraction --library transformers --normalize-embeddings
```

## Config Files

Useful example files:

- `docs/batch_presets_example.yaml`
- `docs/batch_commands.txt`
- `src/model_exporter/config/policy.yaml`
