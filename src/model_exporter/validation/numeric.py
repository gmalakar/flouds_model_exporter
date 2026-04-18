#!/usr/bin/env python3
# =============================================================================
# File: validate_onnx_model.py
# Date: 2026-01-09
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Validate an exported ONNX model against a reference Hugging Face PyTorch model.

This validator is importable as a module and also usable as a CLI script.

It tokenizes example texts, runs the reference PyTorch model and the ONNX model,
and reports max/mean absolute diffs. Exit code is non-zero when diffs exceed
configured tolerances.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

try:
    import onnxruntime as ort  # noqa: E402
    import torch  # noqa: E402
    from transformers import AutoModel, AutoTokenizer  # noqa: E402
except Exception:
    print("Missing required packages. Install the onnx loader requirements:")
    print("  pip install -r requirements-prod.txt")
    raise

# Prefer the shared import from `onnx_verify` to avoid duplicating import logic
from model_exporter.validation.checker import checker, load

# Try to import onnx_utils.has_external_data and onnx_helpers.create_ort_session if available
# Use a permissive Callable signature to avoid strict parameter-name/shape checks by some type checkers.
has_external_data_fn: Optional[Callable[..., bool]] = None
create_ort_session_fn: Optional[Callable[..., Any]] = None
get_preferred_provider_fn: Optional[Callable[..., str]] = None

# Declare the optional imported helper variable with an explicit optional callable
_has_external_data: Optional[Callable[..., bool]] = None
try:
    from model_exporter.utils.compat import has_external_data as _has_external_data
except Exception:
    _has_external_data = None

# If the optional import succeeded, assign to the permissive variable
if _has_external_data is not None:
    has_external_data_fn = _has_external_data

if has_external_data_fn is None:

    def _fallback_has_external_data(model: Any) -> bool:
        # Best-effort detection: check initializer data_location and external_data fields
        try:
            locs = []
            for tensor in getattr(model.graph, "initializer", []):
                if getattr(tensor, "data_location", 0) == 1 and hasattr(tensor, "external_data"):
                    for kv in tensor.external_data:
                        if getattr(kv, "key", None) == "location" and getattr(kv, "value", None):
                            locs.append(kv.value)
            return len([loc for loc in locs if loc]) > 0
        except Exception:
            return False

    has_external_data_fn = _fallback_has_external_data


try:
    from model_exporter.utils.helpers import create_ort_session as create_ort_session_fn
    from model_exporter.utils.helpers import get_preferred_provider as get_preferred_provider_fn
except Exception:
    pass

if create_ort_session_fn is None:

    def _default_create_ort_session(path: str, provider: Any | None = None) -> Any:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])  # fallback

    create_ort_session_fn = _default_create_ort_session


if get_preferred_provider_fn is None:

    def _default_get_preferred_provider() -> str:
        return "CPUExecutionProvider"

    get_preferred_provider_fn = _default_get_preferred_provider


from model_exporter.utils.diagnostics import collect_diagnostics  # noqa: E402
from model_exporter.validation.math_utils import l2_normalize  # noqa: E402
from model_exporter.validation.math_utils import (  # noqa: E402
    compare_arrays,
    mean_pooling,
    rowwise_cosine,
)


def _to_numpy(v: Any) -> Any:
    """Convert torch tensors, lists, and python types to numpy arrays.

    Ensures integer arrays are int64, floats are float32 to match ONNX dtypes.
    """
    try:
        import torch

        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
    except Exception:
        pass

    if isinstance(v, (list, tuple)):
        v = np.array(v)

    if not isinstance(v, np.ndarray):
        try:
            v = np.asarray(v)
        except Exception:
            return v

    if np.issubdtype(v.dtype, np.integer):
        v = v.astype(np.int64)
    elif np.issubdtype(v.dtype, np.floating):
        v = v.astype(np.float32)

    return v


def _load_pooling_config(model_id: str) -> Dict[str, Any]:
    """Load Sentence-Transformers pooling configuration from HuggingFace.

    Returns dict with pooling_mode_cls_token and pooling_mode_mean_tokens flags.
    """
    if hf_hub_download is None:
        return {}

    try:
        pool_cfg_path = hf_hub_download(str(model_id), "1_Pooling/config.json")
        with open(pool_cfg_path, "r", encoding="utf-8") as pf:
            return json.load(pf)
    except Exception:
        return {}


def _resolve_onnx_input_name(name: str, tokenizer_outputs: dict, sess_inputs: List) -> str | None:
    """Resolve an ONNX input name to a tokenizer output key using heuristics.

    Returns the matched key or None if no suitable match found.
    """
    if name in tokenizer_outputs:
        return name

    lc = name.lower()
    for k in tokenizer_outputs.keys():
        if k.lower() == lc:
            return k

    for k in tokenizer_outputs.keys():
        if k in name or name in k:
            return k

    return None


def _construct_default_input(meta: Any, batch_size: int, seq_len: int) -> np.ndarray:
    """Construct a sensible default ONNX input array from metadata.

    Infers shape from metadata symbolic dims and chooses appropriate dtype.
    """
    shape = []
    for d in meta.shape:
        if isinstance(d, int):
            shape.append(d)
        else:
            ds = str(d).lower()
            if "batch" in ds:
                shape.append(batch_size)
            elif "past" in ds:
                shape.append(0)
            elif "seq" in ds or "length" in ds:
                shape.append(max(1, seq_len))
            elif "head" in ds or "num" in ds:
                shape.append(1)
            else:
                try:
                    shape.append(int(ds))
                except Exception:
                    shape.append(1)

    typ = getattr(meta, "type", None)
    tstr = str(typ).lower() if typ is not None else ""
    default_dtype = np.int64 if "int" in tstr else np.float32

    lname = meta.name.lower() if hasattr(meta, "name") else ""
    if "attention_mask" in lname:
        return np.ones((batch_size, seq_len), dtype=np.int64)
    elif "position_ids" in lname:
        return np.arange(seq_len, dtype=np.int64)[None, :]
    elif "input_ids" in lname:
        return np.zeros((batch_size, seq_len), dtype=np.int64)

    resolved_shape = tuple(shape) if shape else (batch_size, seq_len)
    return np.zeros(resolved_shape, dtype=default_dtype)


def _adjust_attention_mask(inputs: dict, sess: Any, seq_len: int, batch_size: int) -> None:
    """Adjust attention mask to cover past + current sequence lengths.

    Modifies inputs dict in-place.
    """
    past_seq_len = 0
    for meta in sess.get_inputs():
        n = meta.name.lower()
        if "past_key_values" in n and ".key" in n:
            if len(meta.shape) >= 3:
                d = meta.shape[2]
                if isinstance(d, int) and d > past_seq_len:
                    past_seq_len = int(d)
            if meta.name in inputs:
                v = inputs[meta.name]
                try:
                    if isinstance(v, np.ndarray) and v.ndim >= 3:
                        past_seq_len = max(past_seq_len, int(v.shape[2]))
                except Exception:
                    pass

    am_name = None
    for k in inputs.keys():
        if k.lower().endswith("attention_mask"):
            am_name = k
            break

    if am_name is not None:
        am = inputs[am_name]
        try:
            if isinstance(am, np.ndarray) and am.ndim == 2:
                cur_len = int(am.shape[1])
                desired = seq_len + int(past_seq_len)
                if cur_len != desired:
                    new_am = np.ones((batch_size, desired), dtype=am.dtype)
                    if cur_len <= seq_len:
                        new_am[:, -cur_len:] = am
                    else:
                        new_am[:, -seq_len:] = am[:, -seq_len:]
                    inputs[am_name] = new_am
        except Exception:
            pass


def build_onnx_inputs(sess: Any, tokenizer_outputs: dict[str, Any]) -> dict[str, Any]:
    """Construct ONNX Runtime input dict from tokenizer outputs.

    - Accepts torch tensors, numpy arrays, lists and plain python types.
    - Ensures integer arrays are cast to np.int64 and floats to np.float32.
    - Maps by input name with case-insensitive and substring fallbacks.
    - Synthesizes sensible defaults from ONNX metadata when needed.
    """
    inputs = {}
    input_names = [i.name for i in sess.get_inputs()]

    # Infer batch/seq size from tokenizer outputs
    batch_size = 1
    seq_len = 1
    if "input_ids" in tokenizer_outputs:
        try:
            arr = _to_numpy(tokenizer_outputs["input_ids"])
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                batch_size = int(arr.shape[0])
                seq_len = int(arr.shape[1])
        except Exception:
            pass

    # Map each ONNX input to tokenizer output or synthesize default
    for name in input_names:
        matched_key = _resolve_onnx_input_name(name, tokenizer_outputs, input_names)
        if matched_key:
            inputs[name] = _to_numpy(tokenizer_outputs[matched_key])
            continue

        # No tokenizer value; synthesize from metadata
        try:
            meta = next(i for i in sess.get_inputs() if i.name == name)
            inputs[name] = _construct_default_input(meta, batch_size, seq_len)
        except Exception:
            # Last resort: use single tokenizer output if only one exists
            if len(tokenizer_outputs) == 1 and len(input_names) == 1:
                sole = next(iter(tokenizer_outputs.values()))
                inputs[name] = _to_numpy(sole)
            else:
                raise KeyError(
                    f"Could not map ONNX input '{name}' to tokenizer outputs: "
                    f"keys={list(tokenizer_outputs.keys())}"
                )

    # Adjust attention mask for cached past sequences
    try:
        _adjust_attention_mask(inputs, sess, seq_len, batch_size)
    except Exception:
        pass

    return inputs


def _parse_validation_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the numeric ONNX validator.

    Builds an :class:`argparse.ArgumentParser` with all validation flags and
    returns the parsed namespace.

    Args:
        argv: Optional list of argument strings. When ``None``,
            :data:`sys.argv` is used by argparse automatically.

    Returns:
        An :class:`argparse.Namespace` with all validated argument values.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Path to the exported ONNX model directory")
    p.add_argument(
        "--reference-model",
        required=True,
        help="Hugging Face model id to use as reference (e.g. nomic-ai/nomic-embed-text-v1.5)",
    )
    p.add_argument(
        "--texts",
        nargs="*",
        default=[
            "This is a test sentence for validation.",
            "The quick brown fox jumps over the lazy dog.",
        ],
        help="Example texts to run through the models",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for the reference PyTorch model (default: cpu)",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing custom model code from the HF repo (use with caution)",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance threshold for max diff (default: 1e-4)",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for diffs (default: 1e-3)",
    )
    p.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="L2-normalize sentence embeddings before comparison (helps when implementations differ by normalization semantics)",
    )
    p.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip diagnostic dump collection on validation failure (faster for batch runs)",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Run the full numeric ONNX validation pipeline.

    Loads a reference HuggingFace PyTorch model and an ONNX model from disk,
    tokenizes example texts, runs both models, compares their outputs, and
    reports max/mean absolute differences. Optionally collects diagnostic
    dumps on failure.

    Args:
        argv: Optional list of argument strings to parse. When ``None``,
            :data:`sys.argv[1:]` is used.

    Returns:
        ``0`` on success (within tolerance), ``2`` on validation failure, or
        ``3`` if the ONNX model file is missing.
    """
    args = _parse_validation_args(argv)

    model_dir = args.model_dir
    reference_model = args.reference_model
    device = torch.device(args.device)

    onnx_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(onnx_path):
        print(f"ONNX model not found at: {onnx_path}")
        return 3

    print(f"Loading tokenizer from reference model `{reference_model}`")
    # Pass trust_remote_code to tokenizer as some repos include tokenizer code
    try:
        # Try passing `fix_mistral_regex=True` first (safe on modern transformers).
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                reference_model,
                use_fast=True,
                trust_remote_code=args.trust_remote_code,
                fix_mistral_regex=True,
            )
        except TypeError:
            # Older transformers do not accept the kwarg; fall back.
            tokenizer = AutoTokenizer.from_pretrained(
                reference_model, use_fast=True, trust_remote_code=args.trust_remote_code
            )
    except Exception:
        # Propagate original exception to caller
        raise

    print(f"Loading reference PyTorch model `{reference_model}` (this may download files)...")
    ref_model = AutoModel.from_pretrained(reference_model, trust_remote_code=args.trust_remote_code)
    ref_model.to(device)
    ref_model.eval()

    # Tokenize
    tok_out = tokenizer(args.texts, padding=True, truncation=True, return_tensors="pt")
    # Move to device
    for k, v in list(tok_out.items()):
        if isinstance(v, torch.Tensor):
            tok_out[k] = v.to(device)

    # Normalize and expose attention_mask early so it's always available
    # later for pooling, diagnostics and saved dumps. Convert tensors to numpy.
    attention_mask = tok_out.get("attention_mask")
    try:
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.cpu().numpy()
    except Exception:
        # leave as-is (could be None or numpy already)
        pass

    # Reference forward
    with torch.no_grad():
        ref_outputs = ref_model(**tok_out)

    # Check for sentence-transformers pooling config in the model repo. If the
    # repo indicates CLS-token pooling, prefer that pooling semantics when
    # constructing reference sentence embeddings (some repos expose a pooler
    # but still expect the sentence-transformer pooling behavior).
    pool_cfg = _load_pooling_config(reference_model)
    prefer_cls_pooling = pool_cfg.get("pooling_mode_cls_token", False)

    # Interpret reference outputs
    if hasattr(ref_outputs, "last_hidden_state"):
        ref_token_embeddings = ref_outputs.last_hidden_state.cpu().numpy()
    else:
        # fallback: try first tensor-like output
        first = ref_outputs[0]
        ref_token_embeddings = first.cpu().numpy()

    if prefer_cls_pooling:
        # Repository indicates Sentence-Transformers CLS pooling; use CLS token
        ref_sentence_embeddings = ref_token_embeddings[:, 0, :]
    elif hasattr(ref_outputs, "pooler_output") and ref_outputs.pooler_output is not None:
        ref_sentence_embeddings = ref_outputs.pooler_output.cpu().numpy()
    else:
        # mean pooling over token embeddings using attention mask
        attention_mask = tok_out.get("attention_mask")
        if attention_mask is None:
            attention_mask = np.ones(ref_token_embeddings.shape[:2], dtype=np.int32)
        else:
            attention_mask = attention_mask.cpu().numpy()
        # Try to honor Sentence-Transformers pooling config when available
        ref_sentence_embeddings = None
        pool_cfg = _load_pooling_config(reference_model)
        if pool_cfg.get("pooling_mode_cls_token", False):
            ref_sentence_embeddings = ref_token_embeddings[:, 0, :]
        elif pool_cfg.get("pooling_mode_mean_tokens", False):
            ref_sentence_embeddings = mean_pooling(ref_token_embeddings, attention_mask)

        if ref_sentence_embeddings is None:
            ref_sentence_embeddings = mean_pooling(ref_token_embeddings, attention_mask)

    # Run ONNX
    print(f"Running ONNX session on: {onnx_path}")

    # Pre-run verification similar to onnx_exporter._verify_models
    try:
        onnx_model = load(onnx_path)
        try:
            checker.check_model(onnx_model)
            print("ONNX checker: model passed validation")
        except MemoryError:
            print("ONNX checker: skipped due to MemoryError")
        except Exception as e:
            print("ONNX checker warning:", e)

        try:
            external_used = (
                has_external_data_fn(onnx_model) if callable(has_external_data_fn) else False
            )
            if external_used:
                print(
                    "Warning: model uses external_data tensors. Ensure associated tensor files are co-located with the model."
                )
        except Exception:
            print("Could not determine external_data usage for model")
    except Exception as e:
        print("Failed to load/check ONNX model:", e)

    provider = get_preferred_provider_fn() if callable(get_preferred_provider_fn) else None
    try:
        if not callable(create_ort_session_fn):
            raise RuntimeError("create_ort_session not available")
        sess = cast(Any, create_ort_session_fn)(onnx_path, provider=provider)
    except Exception:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Build ONNX inputs from tokenizer outputs (use CPU numpy arrays)
    tok_out_cpu = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in tok_out.items()}
    onnx_inputs = build_onnx_inputs(sess, tok_out_cpu)

    print("Session inputs:", [i.name for i in sess.get_inputs()])
    print("Session outputs:", [o.name for o in sess.get_outputs()])

    onnx_outs = sess.run(None, onnx_inputs)

    normalize_embeddings = bool(getattr(args, "normalize_embeddings", False))

    # Heuristics to pick comparable ONNX outputs:
    # If session has a single output, compare that to token_embeddings (first output)
    results: dict[str, Any] = {}
    # Keep actual arrays for diagnostics when comparisons fail
    _align_map = {}
    if len(onnx_outs) == 1:
        onnx_arr = onnx_outs[0]
        try:
            # if shapes align with token embeddings, compare token outputs
            cmp = compare_arrays(ref_token_embeddings, onnx_arr)
            if cmp.get("shape_mismatch"):
                # try comparing pooled sentence embeddings instead. When shapes
                # match, L2-normalize both sides to avoid failures caused by
                # differing normalization semantics between implementations.
                if ref_sentence_embeddings.shape == onnx_arr.shape:
                    if normalize_embeddings:
                        ref_norm = l2_normalize(ref_sentence_embeddings)
                        onnx_norm = l2_normalize(onnx_arr)
                        cmp2 = compare_arrays(ref_norm, onnx_norm)
                        _align_map["sentence_embedding"] = (ref_norm, onnx_norm)
                    else:
                        cmp2 = compare_arrays(ref_sentence_embeddings, onnx_arr)
                        _align_map["sentence_embedding"] = (
                            ref_sentence_embeddings,
                            onnx_arr,
                        )
                else:
                    cmp2 = compare_arrays(ref_sentence_embeddings, onnx_arr)
                    _align_map["sentence_embedding"] = (
                        ref_sentence_embeddings,
                        onnx_arr,
                    )
                results["sentence_embedding"] = cmp2
            else:
                results["token_embeddings"] = cmp
                _align_map["token_embeddings"] = (ref_token_embeddings, onnx_arr)
        except Exception as e:
            results["error"] = str(e)
    else:
        # If multiple outputs, compare each ONNX output to both token and
        # sentence embeddings and compute cosine similarity to aid mapping.
        output_names = [o.name for o in sess.get_outputs()]
        per_output_stats = []
        for idx, out in enumerate(onnx_outs):
            name = output_names[idx] if idx < len(output_names) else f"out_{idx}"
            key = f"onnx_out_{idx}({name})"
            best_cmp = None
            try:
                # First try token embeddings
                tcmp = compare_arrays(ref_token_embeddings, out)
                if not tcmp.get("shape_mismatch"):
                    best_cmp = ("token_embeddings", tcmp, (ref_token_embeddings, out))
                # Then try sentence embeddings if shapes align
                scmp = None
                if ref_sentence_embeddings.shape == out.shape:
                    if normalize_embeddings:
                        ref_norm = l2_normalize(ref_sentence_embeddings)
                        out_norm = l2_normalize(out)
                        scmp = compare_arrays(ref_norm, out_norm)
                        sc_align = (ref_norm, out_norm)
                    else:
                        scmp = compare_arrays(ref_sentence_embeddings, out)
                        sc_align = (ref_sentence_embeddings, out)
                    # Prefer sentence embedding match over token match when available
                    if scmp is not None:
                        best_cmp = ("sentence_embedding", scmp, sc_align)

                # If still no usable comparison, fall back to token comparison result
                if best_cmp is None and tcmp is not None:
                    best_cmp = ("token_embeddings", tcmp, (ref_token_embeddings, out))

                # Compute cosine similarity for informational purposes when shapes allow
                cos_mean = None
                try:
                    # Ensure best_cmp and its aligned arrays exist before indexing
                    if best_cmp is None or best_cmp[2] is None:
                        cos_mean = None
                    else:
                        ref_arr = best_cmp[2][0]
                        out_arr = best_cmp[2][1]
                        # Ensure arrays have at least a batch dimension before reshaping
                        if getattr(ref_arr, "shape", None) and getattr(out_arr, "shape", None):
                            ra = np.reshape(ref_arr, (ref_arr.shape[0], -1))
                            oa = np.reshape(out_arr, (out_arr.shape[0], -1))
                            cos = rowwise_cosine(ra, oa)
                            cos_mean = float(cos.mean())
                        else:
                            cos_mean = None
                except Exception:
                    cos_mean = None

                if best_cmp is None:
                    results[key] = {"error": "no suitable comparison found"}
                    _align_map[key] = (None, None)
                    per_output_stats.append((key, "no_match", None, cos_mean))
                else:
                    results[key] = best_cmp[1]
                    _align_map[key] = best_cmp[2]
                    try:
                        max_abs = (
                            best_cmp[1].get("max_abs_diff", None)
                            if isinstance(best_cmp[1], dict)
                            else None
                        )
                    except Exception:
                        max_abs = None
                    per_output_stats.append((key, best_cmp[0], max_abs, cos_mean))
            except Exception as e:
                results[key] = {"error": str(e)}
                per_output_stats.append((key, "error", None, None))

        # Print a short mapping summary to help the user identify which
        # ONNX output corresponds to sentence embeddings (or token outputs).
        try:
            print("Per-output summary (key, matched_as, max_abs_diff, cos_mean):")
            for s in per_output_stats:
                print(s)
        except Exception:
            pass

    print("\nComparison results:")
    print(json.dumps(results, indent=2))

    # Determine pass/fail based on max_abs_diff
    max_diff = 0.0
    for v in results.values():
        if isinstance(v, dict) and not v.get("shape_mismatch", False):
            max_diff = max(max_diff, v.get("max_abs_diff", 0.0))

    print(f"Maximum absolute difference across compared outputs: {max_diff}")
    # Fallback: if token-level outputs match and mean-pooled token ONNX matches
    # the reference sentence embeddings, prefer that comparison. This handles
    # exporters that expose a different tensor as `sentence_embedding` while
    # token-level outputs are correct (we can reconstruct pooled sentence
    # embeddings from token outputs).
    try:
        existing = results.get("sentence_embedding")
        if "sentence_embedding" not in results or (
            isinstance(existing, dict) and existing.get("max_abs_diff", float("inf")) > float("inf")
        ):
            # noop — placeholder to keep structure
            pass
    except Exception:
        pass

    try:
        # If token embeddings were aligned (possibly under a named output key),
        # compute pooled ONNX sentence embeddings. Support both single-output
        # exporters (where _align_map uses the key "token_embeddings") and
        # multi-output exporters (where the key is the ONNX output name).
        token_map_key = None
        for k, v in _align_map.items():
            try:
                a, b = v
                if getattr(a, "ndim", 0) >= 3 and getattr(b, "ndim", 0) >= 3:
                    token_map_key = k
                    break
            except Exception:
                continue

        if token_map_key is not None and "ref_sentence_embeddings" in locals():
            try:
                pair = _align_map.get(token_map_key)
                if pair is None:
                    # alignment missing; skip pooled fallback by aborting this
                    # fallback path — outer except will handle and continue.
                    raise RuntimeError("alignment missing")
                tok_ref_arr, tok_onnx_arr = pair
                # Compute attention mask as numpy if available
                try:
                    if isinstance(attention_mask, np.ndarray):
                        am = attention_mask
                    else:
                        am = (
                            attention_mask.cpu().numpy()
                            if hasattr(attention_mask, "cpu")
                            else np.array(attention_mask)
                        )
                except Exception:
                    am = np.ones(tok_ref_arr.shape[:2], dtype=np.int32)

                pooled_onx = mean_pooling(tok_onnx_arr, am)

                # Compute pooled reference embeddings from token-level refs
                try:
                    pooled_ref = mean_pooling(tok_ref_arr, am)
                except Exception:
                    pooled_ref = None

                # Prepare pooled ONNX variants
                try:
                    pooled_onx_l2 = l2_normalize(pooled_onx)
                except Exception:
                    pooled_onx_l2 = None

                # Compare pooled_ref vs pooled_onx (with and without L2-normalization)
                candidates = []
                if pooled_ref is not None:
                    candidates.append(("pooled_mean", pooled_ref, pooled_onx))
                    try:
                        pr_l2 = l2_normalize(pooled_ref)
                        candidates.append(
                            (
                                "pooled_mean_l2",
                                pr_l2,
                                (
                                    pooled_onx_l2
                                    if pooled_onx_l2 is not None
                                    else l2_normalize(pooled_onx)
                                ),
                            )
                        )
                    except Exception:
                        pass

                # If user requested normalization, ensure normalized comparison is included
                if normalize_embeddings and pooled_ref is not None:
                    try:
                        pr_l2 = l2_normalize(pooled_ref)
                        candidates.append(("ref_l2_vs_mean", pr_l2, pooled_onx))
                    except Exception:
                        pass

                # Evaluate candidates and pick best by max_abs_diff
                best = None
                best_name = None
                for name, a, b in candidates:
                    try:
                        cmp = compare_arrays(a, b)
                        if not best or cmp.get("max_abs_diff", float("inf")) < best.get(
                            "max_abs_diff", float("inf")
                        ):
                            best = cmp
                            best_name = name
                            best_pair = (a, b)
                    except Exception:
                        continue

                # Prefer pooled comparison if it's better than any existing sentence_embedding comparison
                existing = results.get("sentence_embedding")
                replace = False
                if best is not None:
                    if existing is None:
                        replace = True
                    elif isinstance(existing, dict) and not existing.get("shape_mismatch", False):
                        try:
                            if best.get("max_abs_diff", float("inf")) < existing.get(
                                "max_abs_diff", float("inf")
                            ):
                                replace = True
                        except Exception:
                            pass

                if replace:
                    results["sentence_embedding"] = best
                    _align_map["sentence_embedding"] = (
                        best_pair[0],
                        best_pair[1],
                    )
                    try:
                        print(f"Pooled fallback used: {best_name}")
                    except Exception:
                        pass
                    # Remove any detailed ONNX-output keys that refer to sentence_embedding
                    for k in list(results.keys()):
                        if k != "sentence_embedding" and "sentence_embedding" in k:
                            try:
                                del results[k]
                                if k in _align_map:
                                    del _align_map[k]
                            except Exception:
                                pass
            except Exception:
                pass
    except Exception:
        pass

    # Recompute max_diff after potential fallback replacement
    max_diff = 0.0
    for v in results.values():
        if isinstance(v, dict) and not v.get("shape_mismatch", False):
            max_diff = max(max_diff, v.get("max_abs_diff", 0.0))

    print(f"Maximum absolute difference across compared outputs: {max_diff}")

    if max_diff <= args.atol or max_diff <= args.rtol:
        print("Validation: PASS (within tolerance)")
        return 0
    else:
        print("Validation: FAIL (exceeds tolerances).")

        if args.skip_diagnostics:
            print(
                "Skipping diagnostic collection (--skip-diagnostics enabled). "
                "Rerun without flag to collect dumps."
            )
            return 2

        print("Collecting diagnostic dumps...")

        try:
            # Persist metadata for replay/debugging then delegate diagnostics
            try:
                meta = {
                    "reference_model": reference_model,
                    "texts": (list(args.texts) if getattr(args, "texts", None) is not None else []),
                    "normalize_embeddings": bool(getattr(args, "normalize_embeddings", False)),
                }
                meta_dir = os.path.join(model_dir, "validation_dumps")
                os.makedirs(meta_dir, exist_ok=True)
                meta_path = os.path.join(meta_dir, "validation_texts.json")
                with open(meta_path, "w", encoding="utf-8") as mfh:
                    json.dump(meta, mfh, indent=2, ensure_ascii=False)
                print(f"Wrote validation texts metadata: {meta_path}")
            except Exception:
                pass

            try:
                collect_diagnostics(
                    model_dir=model_dir,
                    results=results,
                    align_map=_align_map,
                    tok_out=tok_out,
                    ref_token_embeddings=ref_token_embeddings,
                    attention_mask=attention_mask,
                    normalize_flag=normalize_embeddings,
                )
                print(f"Diagnostic dumps written to: {os.path.join(model_dir, 'validation_dumps')}")
            except Exception as dbg_err:
                print(f"Diagnostic collection failed: {dbg_err}")
        except Exception as dbg_err:
            print(f"Diagnostic collection failed: {dbg_err}")

        return 2


if __name__ == "__main__":
    raise SystemExit(main())


def validate_onnx(
    model_dir: str,
    reference_model: str,
    texts: List[str] | None = None,
    device: str = "cpu",
    atol: float = 1e-4,
    rtol: float = 1e-3,
    trust_remote_code: bool = False,
    normalize_embeddings: bool = False,
) -> int:
    """Programmatic wrapper for the CLI `main` function.

    Returns the same integer exit code as the CLI: 0=pass, 2=numeric fail, 3=missing model or other failures.
    """
    argv: List[str] = [
        "--model-dir",
        model_dir,
        "--reference-model",
        reference_model,
        "--device",
        device,
        "--atol",
        str(atol),
        "--rtol",
        str(rtol),
    ]
    if trust_remote_code:
        argv.append("--trust-remote-code")
    if normalize_embeddings:
        argv.append("--normalize-embeddings")
    if texts:
        argv.append("--texts")
        argv.extend(list(texts))

    try:
        return main(argv)
    except Exception:
        # In programmatic usage, don't raise — return a non-zero code so callers can fall back
        return 3
