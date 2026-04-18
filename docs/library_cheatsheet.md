# Library Cheatsheet

Use this guide to identify the libraries that are relevant to FloudsModelExporter and how they are used.

## Runtime Dependencies

- `transformers`: model, tokenizer, and config loading.
- `torch`: reference model execution and some fallback export paths.
- `optimum`: primary ONNX export orchestration.
- `optimum-onnx`: Optimum ONNX integration for export and runtime workflows.
- `onnx`: model graph loading, inspection, and file rewriting.
- `onnxruntime`: runtime validation, session creation, and quantization helpers.
- `onnxconverter-common`: float16 conversion helpers.
- `numpy`: array comparison, diagnostics, and validation math.
- `sentence-transformers`: some embedding-model export scenarios.
- `sentencepiece`: tokenizer backend required by some Hugging Face models.
- `safetensors`: safe tensor weight loading.
- `einops`: tensor reshaping helpers used by some model families.
- `accelerate`: compatibility support for large-model loading paths.
- `huggingface_hub`: model download, metadata lookup, and authentication.
- `psutil`: RAM inspection and memory-aware export decisions.
- `fsspec`: filesystem abstraction used by model-loading stacks.
- `protobuf`: ONNX and transformer serialization dependencies.
- `PyYAML`: batch preset and export policy loading.

## Development and Test Tooling

- `pytest`: test runner.
- `black`: formatting.
- `isort`: import sorting.
- `flake8`, `flake8-bugbear`, `flake8-comprehensions`: lint rules.
- `mypy`: static typing.
- `bandit`: security linting.
- `pre-commit`: local hook runner.

Anything not listed here was removed because it referenced other repositories or tooling that is not part of this project.
