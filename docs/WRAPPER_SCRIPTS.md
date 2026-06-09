# Wrapper Scripts Guide

Use this guide to run batch exports through the convenience wrappers on Windows, Linux, and macOS.

## Purpose

The wrapper scripts forward to the canonical Python batch CLI and provide:

- Convenience flags for common export overrides
- Optional config file selection
- Optional text-file import into a temporary batch preset
- Consistent invocation on different operating systems

Both wrappers automatically use the repository `.venv` interpreter when it exists. The `-UseVenv` / `--use-venv` flag makes that requirement explicit and fails fast when `.venv` is missing.

Canonical CLI command shape:

```bash
python -m model_exporter.cli.main batch --preset <name> [options]
```

## Windows Wrapper

Script: `run_exports.ps1`

### Parameters

- `-UseVenv`: require `./.venv/Scripts/python.exe`
- `-Force`: apply `--force` to each batch export
- `-Optimize`: apply `--optimize`
- `-OptimizationLevel <0|1|2|99>`: apply `--optimization-level`; requires `-Optimize`
- `-Cleanup`: apply `--cleanup`
- `-SkipValidator`: apply `--skip-validator` unless active entries request validator-only options
- `-PruneCanonical`: apply `--prune-canonical`; requires `-Cleanup`
- `-Portable`: apply `--portable`; requires `-Optimize`
- `-FailFast`: apply `--fail-fast` (stop on first failure)
- `-Config <path>`: use a YAML/JSON batch config file
- `-TextFile <path>`: parse text commands and build a temporary batch config
- `-BatchFile <path>`: use YAML/JSON config or text commands, detected by extension
- `-Preset <name>`: preset name in the selected config (default: `recommended`)
- `-MinFreeMemoryGB <int>`: minimum free RAM threshold for batch (default: `1`)
- `-LogToFile`: request per-export log files and print the logfile path

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

```powershell
powershell -ExecutionPolicy Bypass -File .\run_exports.ps1 -BatchFile .\docs\batch_presets_full.yaml -Preset full -Optimize -OptimizationLevel 2 -Cleanup
```

## Linux/macOS Wrapper

Script: `run_exports.sh`

### Parameters

- `--use-venv`: require `./.venv/bin/python`
- `--force`: apply `--force` to each batch export
- `--optimize`: apply `--optimize`
- `--optimization-level <0|1|2|99>`: apply `--optimization-level`; requires `--optimize`
- `--cleanup`: apply `--cleanup`
- `--skip-validator`: apply `--skip-validator` unless active entries request validator-only options
- `--prune-canonical`: apply `--prune-canonical`; requires `--cleanup`
- `--portable`: apply `--portable`; requires `--optimize`
- `--fail-fast`: apply `--fail-fast` (stop on first failure)
- `--config <path>`: use a YAML/JSON batch config file
- `--text-file <path>`: parse text commands and build a temporary batch config
- `--batch-file <path>`: use YAML/JSON config or text commands, detected by extension
- `--preset <name>`: preset name in the selected config (default: `recommended`)
- `--min-free-memory-gb <int>`: minimum free RAM threshold for batch (default: `1`)
- `--log-to-file`: request per-export log files and print the logfile path

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

```bash
./run_exports.sh --batch-file ./docs/batch_presets_full.yaml --preset full --optimize --optimization-level 2 --cleanup
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
- `docs/batch_presets_full.yaml`
- `docs/batch_commands.txt`
- `docs/batch_commands_full.txt`
- `src/model_exporter/config/policy.yaml`
