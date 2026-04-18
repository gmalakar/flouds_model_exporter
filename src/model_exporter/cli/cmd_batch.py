# =============================================================================
# File: cli/cmd_batch.py
# Date: 2026-01-10
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Batch subcommand: argument definitions, policy helpers, and _run_batch."""

from __future__ import annotations

import gc
import os
import time

from model_exporter.cli.cmd_export import _build_export_parser, _run_export

# Default policy path: relative to this file so it works both in dev (src/)
# and when the package is installed.
_default_export_policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "policy.yaml"))


def _add_batch_arguments(parser):
    """Register all batch-specific CLI arguments onto *parser*.

    Adds arguments for config path, preset name, memory thresholds,
    fail-fast behavior, and global overrides (device, force, optimize, etc.)
    that are forwarded to every export entry in the batch.

    Args:
        parser: An :class:`argparse.ArgumentParser` instance to populate.
    """
    parser.add_argument(
        "--config",
        default=_default_export_policy_path,
        help="Path to YAML batch/export policy config (default: model_exporter/config/policy.yaml)",
    )
    parser.add_argument(
        "--preset",
        default="recommended",
        help="Named batch preset to run (default: recommended)",
    )
    parser.add_argument(
        "--min-free-memory-gb",
        type=int,
        default=1,
        help="Minimum free RAM required before each export (default: 1)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop batch execution on first failed export.",
    )
    parser.add_argument(
        "--onnx-path",
        dest="onnx_path",
        default=None,
        help="Optional ONNX output directory override applied to all batch entries.",
    )
    parser.add_argument(
        "--framework",
        dest="framework",
        default=None,
        help="Optional framework override applied to all batch entries.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default=None,
        help="Optional device override applied to all batch entries.",
    )
    parser.add_argument("--force", action="store_true", help="Apply --force to all batch exports.")
    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help="Apply --skip-validator to all batch exports.",
    )
    parser.add_argument("--optimize", action="store_true", help="Apply --optimize to all batch exports.")
    parser.add_argument("--cleanup", action="store_true", help="Apply --cleanup to all batch exports.")
    parser.add_argument(
        "--prune-canonical",
        dest="prune_canonical",
        action="store_true",
        help="Apply --prune-canonical to all batch exports.",
    )
    parser.add_argument(
        "--no-local-prep",
        dest="no_local_prep",
        action="store_true",
        help="Apply --no-local-prep to all batch exports.",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Apply --portable to all batch exports.",
    )
    parser.add_argument(
        "--use-sub-process",
        dest="use_subprocess",
        action="store_true",
        help="Apply --use-sub-process to all batch exports.",
    )
    # Single flag to request logging to file for each export
    parser.add_argument(
        "--log-to-file",
        dest="log_to_file",
        action="store_true",
        default=False,
        help="Log to file and print log file path in terminal (opt-in).",
    )


def _load_export_policy(config_path):
    """Load and parse the YAML export policy config file.

    Args:
        config_path: Filesystem path to a YAML file containing the export
            policy (typically ``export_policy.yaml``).

    Returns:
        A ``dict`` representing the top-level YAML mapping.

    Raises:
        RuntimeError: If PyYAML is not installed, the file is missing, or the
            YAML root is not a mapping.
    """
    try:
        yaml = __import__("yaml")
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for config-driven batch execution. " "Install runtime dependencies from requirements-prod.txt."
        ) from exc

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise RuntimeError(f"Export policy config not found: {config_path}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Export policy config must be a YAML mapping at top level: {config_path}")
    return data


def _get_batch_preset(name, config_path):
    """Return the list of export entries for a named batch preset.

    Args:
        name: The preset key inside ``batch_presets`` in the policy config.
        config_path: Path to the YAML policy file.

    Returns:
        A ``list`` of export-entry dicts belonging to the preset.

    Raises:
        RuntimeError: If the preset is not found or the policy structure is
            invalid.
    """
    policy = _load_export_policy(config_path)
    presets = policy.get("batch_presets", {})
    if not isinstance(presets, dict):
        raise RuntimeError(f"Invalid batch_presets section in export policy config: {config_path}")

    preset = presets.get(name)
    if preset is None:
        available = ", ".join(sorted(presets.keys())) or "<none>"
        raise RuntimeError(f"Unknown batch preset '{name}'. Available presets: {available}")
    if not isinstance(preset, list):
        raise RuntimeError(f"Batch preset '{name}' must be a list of export entries in {config_path}")
    return preset


def _get_memory_status():
    """Return a snapshot of current system RAM usage via *psutil*.

    Returns:
        A ``dict`` with keys ``total_gb``, ``free_gb``, ``used_gb``, and
        ``free_percent``, or ``None`` if *psutil* is unavailable.
    """
    try:
        psutil = __import__("psutil")
        vm = psutil.virtual_memory()
        total_gb = round(vm.total / (1024**3), 2)
        free_gb = round(vm.available / (1024**3), 2)
        used_gb = round(total_gb - free_gb, 2)
        free_percent = round((vm.available / vm.total) * 100, 1)
        return {
            "total_gb": total_gb,
            "free_gb": free_gb,
            "used_gb": used_gb,
            "free_percent": free_percent,
        }
    except Exception:
        return None


def _write_memory_status(context):
    """Print a human-readable RAM status line prefixed with *context*.

    Args:
        context: A short label (e.g. ``"Pre-Export"``) to identify when the
            snapshot was taken.
    """
    mem = _get_memory_status()
    if mem is None:
        return
    print(f"[{context}] RAM: {mem['used_gb']}GB used / {mem['total_gb']}GB total " f"({mem['free_gb']}GB free, {mem['free_percent']}%)")


def _memory_available(min_free_gb):
    """Return ``True`` when at least *min_free_gb* GB of RAM is available.

    Prints a warning and returns ``False`` when the threshold is not met.
    Always returns ``True`` when *psutil* is unavailable.

    Args:
        min_free_gb: Minimum free gigabytes required.

    Returns:
        ``bool`` indicating whether sufficient memory is available.
    """
    mem = _get_memory_status()
    if mem is None:
        return True
    if mem["free_gb"] < min_free_gb:
        print(f"Warning: low memory: only {mem['free_gb']}GB free " f"(threshold: {min_free_gb}GB)")
        return False
    return True


def _invoke_memory_cleanup():
    """Trigger Python garbage collection and briefly sleep to reclaim memory."""
    gc.collect()
    time.sleep(0.1)


def _export_config_to_argv(config):
    """Convert a batch-preset entry ``dict`` to an ``export`` sub-command argv list.

    Translates Python-style keys (e.g. ``model_name``) to their CLI flag
    equivalents (e.g. ``--model-name``), and boolean flags to
    presence/absence of the flag argument.

    Args:
        config: A single export-entry dict from a batch preset.

    Returns:
        A ``list[str]`` suitable for passing to
        :func:`argparse.ArgumentParser.parse_args`.
    """
    value_options = {
        "model_name": "--model-name",
        "model_for": "--model-for",
        "task": "--task",
        "model_folder": "--model-folder",
        "onnx_path": "--onnx-path",
        "framework": "--framework",
        "opset_version": "--opset-version",
        "device": "--device",
        "quantize": "--quantize",
        "pack_single_threshold_mb": "--pack-single-threshold-mb",
        "hf_token": "--hf-token",
        "library": "--library",
    }
    flag_options = {
        "optimize": "--optimize",
        "trust_remote_code": "--trust-remote-code",
        "normalize_embeddings": "--normalize-embeddings",
        "require_validator": "--require-validator",
        "skip_validator": "--skip-validator",
        "force": "--force",
        "pack_single_file": "--pack-single-file",
        "use_external_data_format": "--use-external-data-format",
        "no_local_prep": "--no-local-prep",
        "merge": "--merge",
        "cleanup": "--cleanup",
        "prune_canonical": "--prune-canonical",
        "no_post_process": "--no-post-process",
        "portable": "--portable",
        "use_subprocess": "--use-sub-process",
        "low_memory_env": "--low-memory-env",
        "log_to_file": "--log-to-file",
    }

    argv = []
    for key, option in value_options.items():
        value = config.get(key)
        if value is not None:
            argv.extend([option, str(value)])

    for key, option in flag_options.items():
        if config.get(key):
            argv.append(option)

    return argv


def _run_batch(args, parser):
    """Execute a named batch preset, exporting each model in sequence.

    Applies global overrides (device, force, etc.) from *args* to every
    preset entry, checks memory before each export, and tracks
    success/failure/skipped counts. Stops early when ``--fail-fast`` is
    set and a model export fails.

    Args:
        args: Parsed :class:`argparse.Namespace` from the batch sub-command.
        parser: The root :class:`argparse.ArgumentParser` (used for error
            reporting).
    """
    export_parser = _build_export_parser(add_help=False)
    preset_items = _get_batch_preset(args.preset, args.config)

    print("======================================")
    print(f"Starting batch preset: {args.preset}")
    print("======================================")
    _write_memory_status("Initial")

    total = len(preset_items)
    success_count = 0
    failed_count = 0
    skipped_count = 0

    global_overrides = {
        "onnx_path": args.onnx_path,
        "framework": args.framework,
        "device": args.device,
        "force": args.force,
        "skip_validator": args.skip_validator,
        "optimize": args.optimize,
        "cleanup": args.cleanup,
        "prune_canonical": args.prune_canonical,
        "no_local_prep": args.no_local_prep,
        "portable": args.portable,
        "use_subprocess": args.use_subprocess,
        "log_to_file": args.log_to_file,
    }

    for index, preset in enumerate(preset_items, start=1):
        print(f"\n--- Export {index} of {total} ---")
        _write_memory_status("Pre-Export")

        if preset.get("skip", False):
            print(f"[SKIP] Skipping model {preset.get('model_name', '<unknown>')} (skip: true)")
            skipped_count += 1
            continue

        if not _memory_available(args.min_free_memory_gb):
            print("Warning: insufficient memory for safe export. Performing cleanup...")
            _invoke_memory_cleanup()
            _write_memory_status("Post-Cleanup")
            if not _memory_available(args.min_free_memory_gb):
                print("Error: still insufficient memory after cleanup. Skipping this export.")
                skipped_count += 1
                continue

        merged = dict(preset)
        for key, value in global_overrides.items():
            if isinstance(value, bool):
                if value:
                    merged[key] = value
            elif value is not None:
                merged[key] = value

        export_argv = _export_config_to_argv(merged)
        parsed_args = export_parser.parse_args(export_argv)
        display_cmd = f"export {' '.join(export_argv)}"
        print(f"=== Running: {display_cmd}")

        start_time = time.time()
        try:
            _run_export(parsed_args, export_parser)
            duration = round(time.time() - start_time, 1)
            print(f"[OK] Success ({duration}s): {display_cmd}")
            success_count += 1
        except Exception as exc:
            duration = round(time.time() - start_time, 1)
            print(f"[FAIL] Failed ({duration}s): {display_cmd} - {exc}")
            failed_count += 1
            if args.fail_fast:
                break

        _write_memory_status("Post-Export")
        _invoke_memory_cleanup()
        _write_memory_status("Post-Cleanup")

    print("\n======================================")
    print("Batch preset complete")
    print("======================================")
    print(f"Total:   {total}")
    print(f"Success: {success_count}")
    print(f"Failed:  {failed_count}")
    print(f"Skipped: {skipped_count}")
    _write_memory_status("Final")
