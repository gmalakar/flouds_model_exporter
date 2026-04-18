# Model Export Workflow

Use this guide to run the export workflow, apply the right command patterns, and troubleshoot timeout-related failures.

## Canonical Entry Points

Prefer the installed console script:

```powershell
flouds-export export --help
flouds-export validate --help
flouds-export optimize --help
flouds-export batch --help
```

If the package is not installed in editable mode, use the module entrypoint:

```powershell
python -m model_exporter.cli.main export --help
```

`run_exports.ps1` is only a Windows convenience wrapper around the Python `batch` subcommand.

## Current Package Layout

```text
src/model_exporter/ (also available via `src/model_exporter/` alias)
├── cli/
│   ├── main.py
│   ├── cmd_export.py
│   ├── cmd_validate.py
│   ├── cmd_optimize.py
│   └── cmd_batch.py
├── config/
│   ├── logging.py
│   └── policy.yaml
├── export/
│   ├── pipeline.py
│   ├── optimizer.py
│   ├── helpers.py
│   └── subprocess_runner.py
├── utils/
└── validation/
```

## End-to-End Flow

1. Preparation
   - Resolve the model source from Hugging Face or a local directory.
   - Load tokenizer and config metadata.
   - Optionally prepare a local copy for some LLM exports to stabilize tied weights.

2. Export
   - Primary path uses `optimum.exporters.onnx.main_export`.
   - Export settings are derived from `--model-for`, `--task`, opset, device, and portability flags.
   - The pipeline can retry with safer fallback settings if the first export strategy fails.

3. Verification and validation
   - Structural verification checks expected ONNX artifacts and basic session loadability.
   - Numeric validation compares ONNX outputs against the reference Hugging Face model unless validation is intentionally skipped.

4. Optimization
   - The optimizer rewrites eligible ONNX artifacts in-place.
   - `--portable` caps optimizations to conservative passes for better cross-machine compatibility.

5. Cleanup
   - Optional cleanup removes redundant ONNX artifacts and temporary export files.
   - Validation dumps are removed automatically after successful validation.

## Model-to-Task Mapping

| Model purpose | `--model-for` | Typical task |
| --- | --- | --- |
| Embeddings / feature extraction | `fe` | `feature-extraction` |
| Seq2seq encoder-decoder | `s2s` | `seq2seq-lm` |
| Sequence classification | `sc` | `sequence-classification` |
| Cross-encoder / ranking | `ranker` | `sequence-classification` |
| Decoder-only LLM | `llm` | `text-generation-with-past` |

Notes:

- `--merge` only applies to decoder-only LLM exports.
- Seq2seq models produce `encoder_model.onnx` and `decoder_model.onnx`, and may also produce `decoder_with_past_model.onnx` when the task requests KV-cache export.
- Large models often require `--use-external-data-format` and may benefit from `--use-sub-process` plus `--no-post-process`.

## Common Commands

### Feature extraction

```powershell
flouds-export export --model-name sentence-transformers/all-MiniLM-L6-v2 --model-for fe --task feature-extraction --optimize
```

```powershell
flouds-export export --model-name BAAI/bge-small-en-v1.5 --model-for fe --task feature-extraction --optimize --normalize-embeddings
```

### Ranking

```powershell
flouds-export export --model-name cross-encoder/ms-marco-MiniLM-L-12-v2 --model-for ranker --task sequence-classification --optimize
```

### Seq2seq

Recommended CPU-friendly encoder-decoder export:

```powershell
flouds-export export --model-name t5-small --model-for s2s --task seq2seq-lm --optimize --library transformers
```

KV-cache variant when interactive decoding is needed:

```powershell
flouds-export export --model-name t5-small --model-for s2s --task text2text-generation-with-past --optimize --pack-single-file --library transformers
```

### Decoder-only LLM

```powershell
flouds-export export --model-name microsoft/phi-2 --model-for llm --task text-generation-with-past --optimize --library transformers --trust-remote-code --skip-validator --merge
```

### Batch runs

```powershell
flouds-export batch --preset recommended --optimize --cleanup --portable
```

Custom policy file:

```powershell
flouds-export batch --config src/model_exporter/config/policy.yaml --preset recommended
```

The batch runner reads presets from `src/model_exporter/config/policy.yaml`.

## Output Layout

```text
onnx/models/
├── fe/
├── s2s/
├── sc/
├── ranker/
└── llm/
```

Typical outputs:

- Encoder-only models: `model.onnx`
- Seq2seq models: `encoder_model.onnx`, `decoder_model.onnx`, optionally `decoder_with_past_model.onnx`
- Decoder-only merged LLM exports: `model.onnx`, optionally `.onnx_data` sidecar files for large models

## Large Model Guidance

Use these flags first when the model is large or export memory is tight:

```powershell
flouds-export export --model-name gpt2-large --model-for llm --task text-generation-with-past --use-external-data-format --use-sub-process --no-post-process --opset-version 11
```

Recommended adjustments:

1. Enable `--use-external-data-format` for large protobuf payloads.
2. Use `--use-sub-process` to isolate crashes and memory leaks.
3. Add `--no-post-process` when export post-processing is the part that fails.
4. Lower `--opset-version` to `14` or `11` if the model fails with newer opsets.
5. Use `--portable` if you need safer optimizations across environments.

## Timeout and Hanging Export Behavior

The export subprocess runner uses a one-hour timeout for subprocess-based exports. This prevents export jobs from hanging indefinitely.

Operational impact:

- Long-running subprocess exports terminate instead of hanging forever.
- Timeout failures are surfaced as export errors.
- The PowerShell wrapper delegates to the same Python batch path, so timeout handling stays centralized.

If a model times out:

1. Re-run the export directly with the Python CLI and fewer optional flags.
2. Check available disk space and RAM before retrying.
3. Try `--no-post-process`, `--use-external-data-format`, or a lower opset.
4. If the model legitimately needs more time, adjust the subprocess timeout in `src/model_exporter/export/subprocess_runner.py`.

Example debug run:

```powershell
$env:TRANSFORMERS_VERBOSITY = "debug"
flouds-export export --model-name microsoft/Phi-3.5-mini-instruct --model-for llm --task text-generation-with-past --library transformers --merge
```

## Troubleshooting

| Issue | Recommended response |
| --- | --- |
| `ModuleNotFoundError: optimum` | Install runtime dependencies from `requirements-prod.txt`. |
| `MemoryError` or OOM crash | Use `--use-sub-process`, `--use-external-data-format`, and possibly `--no-post-process`. |
| `RuntimeError` about 2 GiB protobuf size | Use `--use-external-data-format`. |
| Unsupported or failing opset | Retry with `--opset-version 14` or `11`. |
| Validation mismatch | Re-run without optimization first, then compare with `--normalize-embeddings` when applicable. |
| Remote-code requirement | Add `--trust-remote-code` only after reviewing the model repository. |

Logs are written under `logs/onnx_exports/` unless overridden by `FLOUDS_LOG_DIR`.

## Related Docs

- `README.md` for installation and quick start
- `docs/RELEASE_PROCESS.md` for release automation
- `docs/GITHUB_REPOSITORY_SETTINGS.md` for repository governance settings
