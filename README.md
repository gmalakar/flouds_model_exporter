[![PyPI Version](https://img.shields.io/pypi/v/flouds-model-exporter.svg)](https://pypi.org/project/flouds-model-exporter/)
![Python Versions](https://img.shields.io/pypi/pyversions/flouds-model-exporter.svg)
![License](https://img.shields.io/pypi/l/flouds-model-exporter.svg)
![Build](https://github.com/gmalakar/flouds_model_exporter/actions/workflows/publish-pypi.yml/badge.svg)

# flouds_model_exporter

Production-grade ONNX model export toolkit for HuggingFace transformers.

## Overview

flouds_model_exporter provides a unified pipeline for converting HuggingFace models to optimized ONNX format:

- **Universal Export** – Supports embedding models, seq2seq, classification, and large language models (LLMs)
- **Smart Optimization** – Automatic ONNX optimization with configurable levels and portability modes
- **Robust Validation** – Numeric verification ensuring export accuracy before deployment
- **Large Model Support** – External-data format, subprocess isolation, and memory management for multi-GB models
- **Batch Orchestration** – Python-native batch subcommand with YAML-driven presets for automated multi-model export workflows
- **Fallback Strategies** – Automatic opset retry, trust_remote_code handling, and error recovery

## Quick Start

### Installation

#### From PyPI (recommended)

```bash
pip install flouds-model-exporter
```

#### From source

```powershell
# Clone the repository
git clone https://github.com/gmalakar/flouds_model_exporter.git
cd flouds_model_exporter

# Create a Python 3.11 or 3.12 virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the package and all dependencies
pip install -e .

# (Optional) Install developer tooling
pip install -e ".[dev]"
```

After installation the CLI entry point is available:

```powershell
flouds-export export --help
```

### Environment Variables

Use environment variables to control default output location and Hugging Face authentication.

#### ONNX_PATH

`ONNX_PATH` sets the default ONNX output root used by export workflows.

Windows PowerShell (current session):

```powershell
$Env:ONNX_PATH = "C:\path\to\onnx\models"
```

Linux/macOS (current shell):

```bash
export ONNX_PATH="/path/to/onnx/models"
```

#### HUGGINGFACE_TOKEN

`HUGGINGFACE_TOKEN` provides an access token for private/gated Hugging Face model downloads.

Windows PowerShell (current session):

```powershell
$Env:HUGGINGFACE_TOKEN = "hf_xxx_your_token"
```

Linux/macOS (current shell):

```bash
export HUGGINGFACE_TOKEN="hf_xxx_your_token"
```

You can also pass a token directly per command with `--hf-token`.

#### Persisting Variables

Windows (future terminals):

```powershell
setx ONNX_PATH "C:\path\to\onnx\models"
setx HUGGINGFACE_TOKEN "hf_xxx_your_token"
```

Linux/macOS (bash/zsh profile):

```bash
echo 'export ONNX_PATH="/path/to/onnx/models"' >> ~/.bashrc
echo 'export HUGGINGFACE_TOKEN="hf_xxx_your_token"' >> ~/.bashrc
```

#### Verify Values

Windows PowerShell:

```powershell
echo $Env:ONNX_PATH
echo $Env:HUGGINGFACE_TOKEN
```

Linux/macOS:

```bash
echo "$ONNX_PATH"
echo "$HUGGINGFACE_TOKEN"
```

Security note: never commit real tokens to source control. Rotate any exposed token immediately.

### Export a Model

**Embedding model (Feature Extraction):**
```powershell
flouds-export export `
  --model-name sentence-transformers/all-MiniLM-L6-v2 `
  --model-for fe `
  --task feature-extraction `
  --optimize
```

**Seq2seq model (T5, BART):**
```powershell
flouds-export export `
  --model-name t5-small `
  --model-for s2s `
  --task seq2seq-lm `
  --optimize
```

**Ranker model (Cross-Encoder):**
```powershell
flouds-export export `
  --model-name cross-encoder/ms-marco-MiniLM-L-12-v2 `
  --model-for ranker `
  --task sequence-classification `
  --optimize
```

**Large Language Model (with KV-cache):**
```powershell
flouds-export export `
  --model-name deepseek-ai/deepseek-coder-1.3b-instruct `
  --model-for llm `
  --task text-generation-with-past `
  --use-external-data-format `
  --use-sub-process `
  --use-fallback-if-failed `
  --optimize `
  --merge
```

### Batch Export

Export all configured models with optimizations:
```powershell
flouds-export batch --preset recommended --optimize --cleanup --portable
```

Wrapper script reference: see [docs/WRAPPER_SCRIPTS.md](docs/WRAPPER_SCRIPTS.md) for complete parameter documentation for `run_exports.ps1` and `run_exports.sh`.

Windows users can still use `.\run_exports.ps1`, which forwards to the Python CLI batch subcommand.
Batch presets are loaded from `src/model_exporter/config/policy.yaml`, and you can point to a custom YAML file with `--config`.

Linux/macOS users can use `./run_exports.sh` with the same batch concepts:

```bash
chmod +x ./run_exports.sh
./run_exports.sh --config ./docs/batch_presets_example.yaml --preset text-import --fail-fast
```

Note: the `--suppress-warning` wrapper/CLI option has been removed. To control logging behavior use `--log-to-file` (or `-LogToFile` for the PowerShell wrapper) which requests per-export log files and tee'ing of stdout/stderr into the logfile. By default the exporter writes logs to file unless overridden.

#### Batch Examples (YAML and Text File)

YAML preset example file:

- `docs/batch_presets_example.yaml`

Run using YAML preset:

```powershell
.\run_exports.ps1 -Config .\docs\batch_presets_example.yaml -Preset text-import -FailFast
```

Text command list example file:

- `docs/batch_commands.txt`

Run using text file import:

```powershell
.\run_exports.ps1 -TextFile .\docs\batch_commands.txt -Preset text-import -FailFast
```

Note: text file entries must use the new hyphenated CLI flags (for example `--opset-version`, not `--opset_version`).

### Validate An Export

Validate an exported ONNX model against its reference Hugging Face model:

```powershell
flouds-export validate --model-dir onnx/models/fe/all-MiniLM-L6-v2 --reference-model sentence-transformers/all-MiniLM-L6-v2 --normalize-embeddings
```

### Optimize Existing Exported Models

Run the shared optimizer service against an already-exported ONNX directory:

```powershell
flouds-export optimize --model-dir onnx/models/fe/all-MiniLM-L6-v2 --model-for fe --optimization-level 2 --portable
```

## Python API

After installing the package you can call the exporter directly from Python without using the CLI.

### Basic usage

If `ONNX_PATH` is set, you can omit `onnx_path` and the exporter will use it automatically:

```python
import os
os.environ["ONNX_PATH"] = "/path/to/onnx/models"  # or set it before launching Python

from model_exporter.export.pipeline import export

output_dir = export(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_for="fe",
    task="feature-extraction",
    optimize=True,
    # onnx_path not needed — picked up from ONNX_PATH env var
)
print(f"Exported to: {output_dir}")
```

Or pass `onnx_path` explicitly to override the environment variable:

```python
output_dir = export(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_for="fe",
    task="feature-extraction",
    onnx_path="./custom/onnx",  # overrides ONNX_PATH
    optimize=True,
)
print(f"Exported to: {output_dir}")
```

### Seq2seq (T5, BART)

```python
export(
    model_name="t5-small",
    model_for="s2s",
    task="seq2seq-lm",
    optimize=True,
)
```

### Large model with subprocess isolation

```python
export(
    model_name="meta-llama/Llama-2-7b-hf",
    model_for="llm",
    task="text-generation-with-past",
    use_external_data_format=True,
    use_subprocess=True,
  use_fallback_if_failed=True,
    merge=True,
    hf_token="hf_xxx_your_token",  # for gated models
)
```

### API reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | required | HuggingFace model ID or local path |
| `model_for` | `str` | `"fe"` | `fe`, `s2s`, `sc`, `llm`, `ranker` |
| `task` | `str` | `None` | e.g. `feature-extraction`, `seq2seq-lm`, `sequence-classification` |
| `onnx_path` | `str` | `"onnx"` | Output directory |
| `optimize` | `bool` | `False` | Run ONNX optimizer after export |
| `optimization_level` | `int` | `99` | ORT optimization level (0–99) |
| `opset_version` | `int` | auto | ONNX opset version |
| `device` | `str` | `"cpu"` | `cpu` or `cuda` |
| `framework` | `str` | `None` | `pt` or `tf` |
| `trust_remote_code` | `bool` | `False` | Allow custom model code |
| `use_external_data_format` | `bool` | `False` | Split model for >2GB exports |
| `use_subprocess` | `bool` | `None` | Run export in isolated subprocess |
| `use_fallback_if_failed` | `bool` | `False` | Enable legacy fallback only if primary export fails |
| `merge` | `bool` | `False` | Merge decoder artifacts (LLMs) |
| `pack_single_file` | `bool` | `False` | Repack external-data into single file |
| `normalize_embeddings` | `bool` | `False` | L2-normalize before validation |
| `skip_validator` | `bool` | `False` | Skip numeric validation |
| `require_validator` | `bool` | `False` | Fail if validation cannot run |
| `quantize` | any | `False` | Quantization configuration |
| `hf_token` | `str` | `None` | HuggingFace auth token (via `**kwargs`) |

## CLI Reference

### Core Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `--model-name` | `str` | HuggingFace model ID or local path |
| `--model-for` | `fe`, `s2s`, `sc`, `ranker`, `llm` | Model type: embedding, seq2seq, classification, ranker (cross-encoder), or language model |
| `--task` | `str` | Export task: `feature-extraction`, `seq2seq-lm`, `sequence-classification`, `text-generation-with-past`, etc. |
| `--framework` | `pt`, `tf` | Framework: PyTorch or TensorFlow |
| `--device` | `cpu`, `cuda` | Target device |
| `--opset-version` | `11`, `14`, `17`, `18` | ONNX opset version (default: 17) |
| `--trust-remote-code` | flag | Allow custom model code execution |

### Export Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--framework` | `pt` | Framework: `pt` (PyTorch) or `tf` (TensorFlow) |
| `--device` | `cpu` | Target device: `cpu` or `cuda` |
| `--opset-version` | `17` | ONNX opset version (11, 14, or 17) |
| `--trust-remote-code` | `false` | Allow custom model code execution (⚠ security risk) |
| `--force` | `false` | Overwrite existing exports |

### Optimization & Validation

| Parameter | Description |
|-----------|-------------|
| `--optimize` | Enable post-export ONNX optimization |
| `--optimization-level` | Optimization level: 0-99 (default: 99) |
| `--portable` | Use conservative optimizations for cross-platform compatibility |
| `--skip-validator` | Skip numeric validation |
| `--require-validator` | Fail build if validation fails |
| `--normalize-embeddings` | L2-normalize embeddings during validation |

The standalone `optimize` subcommand accepts `--model-dir`, `--model-for`, `--optimization-level`, and `--portable` so you can re-run optimization without repeating export.

### Large Model Options

| Parameter | Description |
|-----------|-------------|
| `--use-external-data-format` | Split model into .onnx + .onnx_data files (for >2GB models) |
| `--use-sub-process` | Run export in isolated subprocess (safer for large models) |
| `--use-fallback-if-failed` | Enable legacy fallback exporter only if primary export fails |
| `--no-post-process` | Skip ONNX post-processing (reduces memory usage) |
| `--pack-single-file` | Repack external-data model into single file during validation |
| `--pack-single-threshold-mb` | Size threshold for repacking (default: 1536 MB) |

### Advanced Options

| Parameter | Description |
|-----------|-------------|
| `--merge` | Merge decoder artifacts for LLMs (with-past only) |
| `--no-local-prep` | Skip local model preparation for LLMs |
| `--cleanup` | Remove temporary/extraneous files post-export |
| `--prune-canonical` | Remove canonical models when merged version exists |
| `--hf-token` | HuggingFace API token for private models |
| `--onnx-path` | Custom output directory (default: `./onnx`) |

## Output Structure

Exported models are organized by type and name:

```
onnx/models/
├── fe/                              # Feature extraction (embeddings)
│   ├── all-MiniLM-L6-v2/
│   │   └── model.onnx
│   └── bge-small-en-v1.5/
│       ├── model.onnx
│       └── model.onnx_data          # External data (if >2GB)
├── s2s/                             # Seq2seq models
│   ├── t5-small/
│   │   ├── encoder_model.onnx
│   │   ├── decoder_model.onnx
│   │   └── decoder_with_past_model.onnx
│   └── bart-large-cnn/
└── llm/                             # Large language models
    ├── deepseek-coder-1.3b-instruct/
    │   ├── model.onnx
    │   ├── model.onnx_data
    │   └── model_merged.onnx        # Merged version (if --merge used)
    └── phi-3-mini-4k-instruct/
```

## Architecture

### Directory Structure

```
src/model_exporter/ (also available via `src/model_exporter/` alias)
├── cli/                            # CLI entrypoints and subcommands
├── config/                         # Logging and batch policy
├── export/                         # Export pipeline, helpers, optimizer, subprocess runner
├── utils/                          # Diagnostics and helper utilities
└── validation/                     # Structural and numeric validation
```

### Export Pipeline

1. **Preparation** – Token setup, model validation, output directory creation
2. **Export** – `optimum.exporters.onnx.main_export` with fallback strategies
3. **Validation** – Structural checks + numeric validation (input/output comparison)
4. **Optimization** – ONNX Runtime optimization passes (optional)
5. **Cleanup** – Remove temporary files, prune redundant artifacts

## Memory Management

### Subprocess Isolation

For large models, use subprocess isolation to prevent parent process crashes:

```powershell
flouds-export export `
  --model-name meta-llama/Llama-2-7b-hf `
  --use-sub-process `
  --use-fallback-if-failed `
  --use-external-data-format
```

### Batch Export Memory Monitoring

The batch subcommand monitors available RAM before each export:

```powershell
# Require at least 4GB free RAM before each export
flouds-export batch --preset recommended --min-free-memory-gb 4
```

### Config-Driven Batch Workflow

The batch runner loads presets from YAML:

```powershell
flouds-export batch --config src/model_exporter/config/policy.yaml --preset recommended
```

Each preset entry maps directly to export CLI arguments, which makes export pipelines deterministic and versionable.

### Large Model Best Practices

For models >2GB:

1. **Enable external data format** – Splits model into .onnx + .onnx_data
2. **Use subprocess isolation** – Prevents memory leaks affecting subsequent exports
3. **Skip post-processing** – Reduces peak memory during export
4. **Lower opset version** – Simplifies optimization (try opset 11)

```powershell
flouds-export export `
  --model-name gpt2-large `
  --use-external-data-format `
  --use-sub-process `
  --use-fallback-if-failed `
  --no-post-process `
  --opset-version 11
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: optimum` | Install runtime dependencies: `pip install -r requirements-prod.txt` |
| `MemoryError` or OOM crashes | Use `--use-sub-process` and `--use-external-data-format`; reduce `--optimization-level` |
| Primary export fails on edge models | Retry with `--use-fallback-if-failed` to enable legacy fallback path |
| `RuntimeError: > 2GiB protobuf` | Enable `--use-external-data-format` |
| `ValueError: Unsupported opset` | Lower `--opset-version` to 14 or 11 |
| `TracerWarning: Converting tensor` | Model tracing limitation (usually safe to ignore) |
| Validation failures | Check numeric precision; try `--skip-validator` for known issues |
| `trust_remote_code required` | Add `--trust-remote-code` flag (review model code first) |

### Export Logs

Logs are saved to `logs/onnx_exports/` with per-model timestamped files. Configure log directory via `FLOUDS_LOG_DIR` environment variable.

## Requirements

- **Python**: 3.11 or 3.12
- **System**: 8GB+ RAM (16GB+ for large models)
- **Dependencies**: See `requirements-prod.txt` (runtime) and `requirements-dev.txt` (development)

## Contributing

See `CONTRIBUTING.md` for contribution workflow and local development checks.
For expected behavior and standards, see `CODE_OF_CONDUCT.md` and `SECURITY.md`.
Maintainer release steps are documented in `docs/RELEASE_PROCESS.md`.

## License

Licensed under the Apache License, Version 2.0 (Apache-2.0). See `LICENSE` for details.
