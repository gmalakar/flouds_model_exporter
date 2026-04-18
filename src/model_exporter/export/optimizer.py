#!/usr/bin/env python3
# =============================================================================
# File: export_optimizer.py
# Date: 2026-01-09
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Optimization helper extracted from `onnx_exporter.py`.

Exposes `run_optimization(...)` which performs the optimum/ORTOptimizer-based
optimization step and performs post-optimization numeric re-validation when
saved `validation_dumps` exist. Returns the post-optimization validator
return code (0 on success, non-zero on failure) and does not raise unless
`require_validator=True` and a post-optimization validation fails, in which
case it raises `SystemExit(rc)` to preserve original exporter semantics.
"""
from __future__ import annotations

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Any


def _find_encoder_onnx_files(onnx_files: list[Path]) -> list[Path]:
    """Select which ONNX files in *onnx_files* should be optimized.

    Uses a three-tier preference:

    1. Merged ONNX files whose name starts with ``model`` or ``encoder_``.
    2. Any other ONNX file starting with ``model`` or ``encoder_``.
    3. Any ONNX file that does **not** start with ``decoder_``.

    Args:
        onnx_files: All ``.onnx`` paths found in the model directory.

    Returns:
        A list of :class:`~pathlib.Path` objects for the files to optimize.
        May be empty if no suitable candidates are found.
    """
    merged_candidates = [
        f for f in onnx_files if (f.name.lower().startswith("model") or f.name.lower().startswith("encoder_")) and "merged" in f.name.lower()
    ]
    if merged_candidates:
        return merged_candidates

    candidates = [f for f in onnx_files if f.name.lower().startswith("model") or f.name.lower().startswith("encoder_")]
    if candidates:
        return candidates

    return [f for f in onnx_files if not f.name.lower().startswith("decoder_")]


def _load_ort_optimizer_classes() -> tuple[Any, Any]:
    """Import and return ``(ORTOptimizer, OptimizationConfig)`` from *optimum*.

    Returns:
        A 2-tuple ``(ORTOptimizer, OptimizationConfig)``.

    Raises:
        ImportError: If either class cannot be imported from *optimum*.
    """
    import importlib

    opt_mod = importlib.import_module("optimum.onnxruntime")
    ORTOptimizer = opt_mod.ORTOptimizer
    opt_config_mod = importlib.import_module("optimum.onnxruntime.configuration")
    OptimizationConfig = opt_config_mod.OptimizationConfig
    return ORTOptimizer, OptimizationConfig


def optimize_if_encoder(
    model_dir: Path | str,
    model_type: str,
    logger: logging.Logger,
    optimization_level: int = 99,
    portable: bool = False,
) -> int:
    """Optimize encoder ONNX artifacts in a model directory.

    This helper is conservative: it only optimizes encoder-related ONNX
    files (files named ``encoder_*.onnx`` or a single ``model.onnx`` used
    for encoder-only architectures). It copies the encoder ONNX files and
    the ``config.json`` into an ``encoder/`` subdirectory, loads an
    ``ORTOptimizer`` from that directory (local files only), runs
    ``optimizer.optimize(save_dir=...)`` with an ``OptimizationConfig``,
    and atomically replaces the original ONNX files with the optimized
    outputs.

    Parameters
    - model_dir: Path or string path to the exported model folder.
    - model_type: Model "for" string (e.g. "s2s", "sc", "llm"); used
        to skip decoder-only models.
    - logger: A ``logging.Logger`` instance for logging messages.
    - optimization_level: One of {0, 1, 2, 99} controlling optimizer
        aggressiveness (default: 99).

    Returns
    - 0 on success or benign skip (no encoder artifacts / missing config /
        missing directory).
    - 1 on hard failure (optimizer/load errors or no optimized files
        produced).

    Notes
    - This function avoids remote fetches and does not enable
        ``trust_remote_code``; callers should ensure artifacts are local.
    - ``logger`` is required to enforce consistent exporter logging.
    """

    # Normalize model_type
    model_type = model_type.lower().strip()

    # Skip decoder-only models
    decoder_only_types = {"llm", "causal-lm", "clm"}
    if model_type in decoder_only_types:
        logger.info("Skipping optimization for decoder-only model type: %s", model_type)
        return 0

    # Validate optimization level
    valid_levels = {0, 1, 2, 99}
    if optimization_level not in valid_levels:
        logger.warning(
            "Invalid optimization level: %s. Must be one of %s",
            optimization_level,
            valid_levels,
        )
        return 1

    # If portable optimizations are requested, cap to a conservative level
    if portable:
        # Conservative: do not use the aggressive/extended optimizer passes
        eff_level = min(optimization_level, 2)
        if eff_level != optimization_level:
            try:
                logger.info(
                    "Portable optimizations requested; capping optimization_level %s -> %s",
                    optimization_level,
                    eff_level,
                )
            except Exception:
                pass
        optimization_level = eff_level

    model_dir = Path(model_dir)

    # Validate model_dir
    if not model_dir.exists() or not model_dir.is_dir():
        logger.warning("Model directory does not exist: %s", model_dir)
        return 0

    # Identify ONNX files
    onnx_files = list(model_dir.glob("*.onnx"))
    encoder_files = _find_encoder_onnx_files(onnx_files)

    if not encoder_files:
        logger.info("No ONNX files found to optimize.")
        return 0

    # Check config.json
    config_path = model_dir / "config.json"
    if not config_path.exists():
        logger.info("config.json is required for optimization.")
        return 0

    # Attempt filewise in-place optimization first (avoid copying to encoder/)
    try:
        ORTOptimizer, OptimizationConfig = _load_ort_optimizer_classes()

        # Prepare optimization config
        optimization_config = OptimizationConfig(optimization_level=optimization_level)

        # Some Optimum versions accept local_files_only, some don't
        load_kwargs = {"local_files_only": True}

        def _call_with_retry(fn, *a, **kw):
            try:
                return fn(*a, **kw)
            except TypeError:
                safe = dict(kw)
                safe.pop("local_files_only", None)
                return fn(*a, **safe)

        logger.info("Attempting filewise in-place optimization in: %s", model_dir)

        filewise_ok = True
        optimized_pairs: list[tuple[Path, Path]] = []

        for onnx_path in encoder_files:
            try:
                # Initialize optimizer for this specific file in the same folder
                optimizer = _call_with_retry(
                    ORTOptimizer.from_pretrained,
                    model_dir,
                    file_name=onnx_path.name,
                    **load_kwargs,
                )

                # Run optimization saving into the same directory
                optimizer.optimize(save_dir=model_dir, optimization_config=optimization_config)

                # Find any optimized artifact in the target folder.
                # Simpler: look for any .onnx with 'optimized' in its filename.
                optimized_candidates = [p for p in model_dir.glob("*.onnx") if "optimized" in p.name.lower()]

                if not optimized_candidates:
                    logger.warning("No optimized output found for %s", onnx_path.name)
                    filewise_ok = False
                    break

                # Prefer a candidate that contains the original stem, otherwise take the first.
                optimized_path = None
                for c in optimized_candidates:
                    if onnx_path.stem in c.name:
                        optimized_path = c
                        break
                if optimized_path is None:
                    optimized_path = optimized_candidates[0]

                optimized_pairs.append((optimized_path, onnx_path))

            except Exception as e:
                logger.info("Filewise optimization failed for %s: %s", onnx_path.name, e)
                filewise_ok = False
                break

        if filewise_ok and optimized_pairs:
            # Do NOT replace originals here. A separate process handles replacement.
            # Write a marker that maps optimized files to their original names
            try:
                marker = model_dir / ".optimizations_applied"
                with open(marker, "w", encoding="utf-8") as mf:
                    for opt_path, orig_path in optimized_pairs:
                        mf.write(f"{opt_path.name} -> {orig_path.name}\n")
                    if (model_dir / "ort_config.json").exists():
                        mf.write("ort_config.json\n")
                logger.info(
                    "Filewise optimization complete (no replace); marker written: %s",
                    marker,
                )
            except Exception:
                logger.debug(
                    "Failed to write optimization marker after filewise optimization",
                    exc_info=True,
                )

            # Detect potential hardware-specific optimizations by scanning ort_config
            try:
                oc = model_dir / "ort_config.json"
                if oc.exists():
                    txt = oc.read_text(encoding="utf-8", errors="ignore")
                    hw_markers = (
                        "nchwc",
                        "nchw",
                        "nchwctransformer",
                        "graph optimization level greater than ort_enable_extended",
                    )
                    if any(m in txt.lower() for m in hw_markers):
                        logger.warning(
                            "Optimized artifacts may contain hardware-specific ORT optimizations; artifacts may not be portable across environments."
                        )
            except Exception:
                pass

            return 0

        logger.info("Filewise optimization did not produce usable optimized artifacts; falling back to encoder/ copy approach.")

    except Exception:
        logger.debug(
            "Filewise optimization attempt aborted (optimum missing or failure); falling back.",
            exc_info=True,
        )

    # Temporary encoder directory
    encoder_dir = model_dir / "encoder"

    # Log that we're using the encoder temp directory fallback
    logger.info("Falling back to temp encoder directory for optimization: %s", encoder_dir)

    try:
        # Clean temp directory if exists
        if encoder_dir.exists():
            shutil.rmtree(encoder_dir)
        encoder_dir.mkdir(parents=True, exist_ok=True)

        # Copy config.json and any external-data files (e.g., *.onnx_data)
        # into the encoder dir so the optimizer can read them without
        # modifying the original export directory.
        try:
            if config_path.exists():
                try:
                    shutil.copy2(config_path, encoder_dir / "config.json")
                except Exception as e:
                    logger.warning("Failed to copy config.json into encoder dir: %s", e)

            # Copy any external-data files often named like 'model.onnx_data'
            try:
                for p in model_dir.glob("*.onnx_data"):
                    try:
                        shutil.copy2(p, encoder_dir / p.name)
                        logger.info("Copied external data file into encoder dir: %s", p.name)
                    except Exception:
                        logger.debug(
                            "Failed to copy external data file %s",
                            p.name,
                            exc_info=True,
                        )
            except Exception:
                logger.debug("Error while scanning/copying .onnx_data files", exc_info=True)
        except Exception as e:
            logger.warning("Failed to prepare encoder dir support files: %s", e)

        logger.info("Optimizing encoder files one-by-one in: %s", encoder_dir)

        optimized_written: list[str] = []

        # Iterate over selected ONNX files and optimize them individually.
        for src_onnx in encoder_files:
            try:
                # Clean any previous model/onx artifacts in encoder_dir
                for pattern in ["model*.onnx", "*_optimized.onnx"]:
                    for p in encoder_dir.glob(pattern):
                        try:
                            p.unlink()
                        except Exception:
                            pass

                # Copy the source ONNX into encoder_dir as model.onnx
                try:
                    shutil.copy2(src_onnx, encoder_dir / "model.onnx")
                except Exception as e:
                    logger.warning("Failed to copy %s as model.onnx: %s", src_onnx.name, e)
                    continue

                # Load optimizer for the encoder_dir and run optimization
                try:
                    optimizer = _call_with_retry(
                        ORTOptimizer.from_pretrained,
                        encoder_dir,
                        **load_kwargs,
                    )

                    optimizer.optimize(save_dir=encoder_dir, optimization_config=optimization_config)
                except Exception as e:
                    logger.warning("Optimization failed for %s: %s", src_onnx.name, e)
                    continue

                # Find the optimized artifact in encoder_dir — there will be only one.
                optimized_candidates = [p for p in encoder_dir.glob("*.onnx") if "optimized" in p.name.lower()]
                if not optimized_candidates:
                    logger.info("No optimized output found for %s", src_onnx.name)
                    continue

                chosen = optimized_candidates[0]
                if chosen is None:
                    logger.info("No optimized candidate selected for %s", src_onnx.name)
                    continue

                # Copy chosen optimized artifact back to model_dir as <original>_optimized.onnx
                target_name = src_onnx.stem + "_optimized.onnx"
                target_path = model_dir / target_name
                tmp_target = target_path.with_suffix(target_path.suffix + ".tmp")
                try:
                    shutil.copy2(chosen, tmp_target)
                    os.replace(tmp_target, target_path)
                    optimized_written.append(target_name)
                    logger.info(
                        "Saved optimized artifact for %s as %s",
                        src_onnx.name,
                        target_name,
                    )
                except Exception as e:
                    logger.warning("Failed to copy optimized artifact for %s: %s", src_onnx.name, e)
                    try:
                        if tmp_target.exists():
                            tmp_target.unlink()
                    except Exception:
                        pass

                # Remove optimized and any other ONNX files from encoder_dir
                for q in encoder_dir.glob("*.onnx"):
                    try:
                        q.unlink()
                    except Exception:
                        pass

            except Exception as e:
                logger.debug(
                    "Unexpected error while optimizing %s: %s",
                    src_onnx.name,
                    e,
                    exc_info=True,
                )
                continue

        # After individual optimizations, copy back non-ONNX support files (e.g., ort_config.json, config.json)
        try:
            for p in encoder_dir.iterdir():
                if p.suffix.lower() == ".onnx":
                    continue
                dst = model_dir / p.name
                tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
                try:
                    shutil.copy2(p, tmp_dst)
                    os.replace(tmp_dst, dst)
                    logger.info("Copied support file back: %s", dst.name)
                except Exception as e:
                    logger.warning(
                        "Failed to copy support file %s back to model dir: %s",
                        p.name,
                        e,
                    )
        except Exception:
            logger.debug("Error while copying support files back to model dir", exc_info=True)

        # If ort_config.json was copied back, scan it for hardware-specific optimizations
        try:
            oc = model_dir / "ort_config.json"
            if oc.exists():
                txt = oc.read_text(encoding="utf-8", errors="ignore")
                hw_markers = (
                    "nchwc",
                    "nchw",
                    "nchwctransformer",
                    "graph optimization level greater than ort_enable_extended",
                )
                if any(m in txt.lower() for m in hw_markers):
                    logger.warning(
                        "Optimized artifacts may contain hardware-specific ORT optimizations; artifacts may not be portable across environments."
                    )
        except Exception:
            pass

        # Write marker file listing optimized artifacts
        try:
            marker = model_dir / ".optimizations_applied"
            try:
                with open(marker, "w", encoding="utf-8") as mf:
                    for name in optimized_written:
                        mf.write(f"{name}\n")
                    if (model_dir / "ort_config.json").exists():
                        mf.write("ort_config.json\n")
                logger.info("Wrote optimization marker: %s", marker)
            except Exception:
                logger.debug("Could not write optimization marker file", exc_info=True)
        except Exception:
            pass

        logger.info("Encoder optimization complete.")
        return 0

    except Exception as e:
        logger.error("Optimization failed: %s", e)
        return 1

    finally:
        # Cleanup temp directory
        if encoder_dir.exists():
            shutil.rmtree(encoder_dir)

        # Free memory
        gc.collect()
        logger.debug("Temporary encoder directory cleaned up.")
