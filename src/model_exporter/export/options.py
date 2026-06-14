# =============================================================================
# File: export/options.py
# Date Created: 2026-06-08
# Date Updated: 2026-06-08
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Typed export option container shared by CLI and batch flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


MODEL_FOR_DEFAULTS: dict[str, dict[str, str]] = {
    "fe": {"task": "feature-extraction", "library": "transformers"},
    "s2s": {"task": "seq2seq-lm", "library": "transformers"},
    "llm": {"task": "text-generation-with-past", "library": "transformers"},
    "ranker": {"task": "sequence-classification", "library": "transformers"},
}


def resolve_model_for_defaults(model_for: str, task: str | None, library: str | None) -> tuple[str, str, str]:
    """Resolve model_for, task, and library using canonical CLI defaults."""
    normalized_model_for = (model_for or "fe").lower()
    defaults = MODEL_FOR_DEFAULTS.get(normalized_model_for)
    if defaults is None:
        expected = ", ".join(MODEL_FOR_DEFAULTS)
        raise ValueError(f"Unknown model_for: {model_for!r}. Expected one of: {expected}")
    return (
        normalized_model_for,
        task or defaults["task"],
        library or defaults["library"],
    )


@dataclass(frozen=True)
class ExportConfig:
    """Normalized export options ready to pass to the public export API."""

    model_name: str
    task: str | None = None
    model_for: str = "fe"
    optimize: bool = False
    optimization_level: int | None = None
    portable: bool = False
    model_folder: str | None = None
    onnx_path: str | None = None
    force: bool = False
    opset_version: int | None = None
    pack_single_file: bool = False
    use_external_data_format: bool = False
    framework: str | None = None
    require_validator: bool = False
    trust_remote_code: bool = False
    normalize_embeddings: bool = False
    skip_validator: bool = False
    device: str = "cpu"
    huggingface_hub_token: str | None = None
    library: str | None = None
    merge: bool = False
    cleanup: bool = False
    prune_canonical: bool = False
    no_post_process: bool = False
    no_local_prep: bool = False
    use_subprocess: bool = False
    use_fallback_if_failed: bool = False
    min_free_memory_gb: float | None = None
    require_sufficient_memory: bool = False
    quantize: str | bool | None = False
    log_to_file: bool = False

    @classmethod
    def from_namespace(cls, args: Any, onnx_path: str | None) -> "ExportConfig":
        """Build an export config from argparse output."""
        model_for, task, library = resolve_model_for_defaults(
            args.model_for,
            args.task,
            args.library,
        )
        return cls(
            model_name=args.model_name,
            model_for=model_for,
            optimize=args.optimize,
            optimization_level=args.optimization_level,
            portable=args.portable,
            model_folder=args.model_folder,
            onnx_path=onnx_path,
            task=task,
            force=args.force,
            opset_version=(args.opset_version if hasattr(args, "opset_version") else None),
            pack_single_file=args.pack_single_file,
            use_external_data_format=args.use_external_data_format,
            framework=args.framework,
            require_validator=args.require_validator,
            trust_remote_code=args.trust_remote_code,
            normalize_embeddings=args.normalize_embeddings,
            skip_validator=args.skip_validator,
            device=args.device,
            huggingface_hub_token=args.huggingface_hub_token,
            library=library,
            merge=args.merge,
            cleanup=args.cleanup,
            prune_canonical=args.prune_canonical,
            no_post_process=args.no_post_process,
            no_local_prep=args.no_local_prep,
            use_subprocess=args.use_subprocess,
            use_fallback_if_failed=args.use_fallback_if_failed,
            min_free_memory_gb=args.min_free_memory_gb,
            require_sufficient_memory=args.require_sufficient_memory,
            quantize=args.quantize,
            log_to_file=args.log_to_file,
        )

    def to_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for ``model_exporter.export.pipeline.export``."""
        data = asdict(self)
        model_for, task, library = resolve_model_for_defaults(
            data["model_for"],
            data["task"],
            data["library"],
        )
        data["model_for"] = model_for
        data["task"] = task
        data["library"] = library
        if data["optimization_level"] is None:
            data["optimization_level"] = 99
        if data["quantize"] is None:
            data["quantize"] = False
        elif data["quantize"] == "both":
            data["quantize"] = True
        return data
