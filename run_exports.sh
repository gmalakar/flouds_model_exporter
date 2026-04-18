#!/usr/bin/env bash
# Optional Linux/macOS convenience wrapper for batch exports.
# Canonical batch entrypoint: flouds-export batch --preset recommended
#
# Usage:
#   ./run_exports.sh
#   ./run_exports.sh --use-venv --force
#   ./run_exports.sh --force --optimize --cleanup --skip-validator --prune-canonical --portable --no-local-prep
#   ./run_exports.sh --config ./docs/batch_presets_example.yaml --preset text-import --fail-fast
#   ./run_exports.sh --text-file ./docs/batch_commands.txt --preset text-import --fail-fast
#   ./run_exports.sh --batch-file ./docs/batch_presets_full.yaml --fail-fast
#
# Flags:
#   --use-venv           Use the repository .venv Python interpreter
#   --force              Append --force to each export command
#   --optimize           Append --optimize to each export command
#   --cleanup            Append --cleanup to each export command
#   --skip-validator     Append --skip-validator to each export command
#   --prune-canonical    Append --prune-canonical to each export command
#   --portable           Append --portable to each export command
#   --no-local-prep      Append --no-local-prep to each export command
#   --fail-fast          Stop batch on first failed export
#   --batch-file PATH    YAML/JSON batch config or text file of export commands
#   --min-free-memory-gb INT  Minimum free memory in GB before each export (default: 1)

set -euo pipefail

USE_VENV=0
FORCE=0
SKIP_VALIDATOR=0
OPTIMIZE=0
CLEANUP=0
PRUNE_CANONICAL=0
NO_LOCAL_PREP=0
PORTABLE=0
FAIL_FAST=0
LOG_TO_FILE=0
CONFIG_PATH=""
BATCH_FILE=""
MIN_FREE_MEMORY_GB=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

usage() {
  sed -n '1,40p' "$0" | sed 's/^# \{0,1\}//'
}

resolve_repo_path() {
  local input_path="$1"
  if [[ -z "$input_path" ]]; then
    printf '%s' ""
    return
  fi
  if [[ "$input_path" = /* ]]; then
    printf '%s' "$input_path"
  else
    printf '%s' "$REPO_ROOT/$input_path"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-venv|-UseVenv)
      USE_VENV=1
      shift
      ;;
    --force|-Force)
      FORCE=1
      shift
      ;;
    --skip-validator|-SkipValidator)
      SKIP_VALIDATOR=1
      shift
      ;;
    --optimize|-Optimize)
      OPTIMIZE=1
      shift
      ;;
    --cleanup|-Cleanup)
      CLEANUP=1
      shift
      ;;
    --prune-canonical|-PruneCanonical)
      PRUNE_CANONICAL=1
      shift
      ;;
    --no-local-prep|-NoLocalPrep)
      NO_LOCAL_PREP=1
      shift
      ;;
    --portable|-Portable)
      PORTABLE=1
      shift
      ;;
    --fail-fast|-FailFast)
      FAIL_FAST=1
      shift
      ;;
    --batch-file|-BatchFile)
      BATCH_FILE="${2:-}"
      shift 2
      ;;
    --min-free-memory-gb|-MinFreeMemoryGB)
      MIN_FREE_MEMORY_GB="${2:-}"
      shift 2
      ;;
    --log-to-file|-LogToFile)
      LOG_TO_FILE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$USE_VENV" -eq 1 ]]; then
  PYTHON_EXE="$REPO_ROOT/.venv/bin/python"
  if [[ ! -x "$PYTHON_EXE" ]]; then
    echo "Python venv not found at $PYTHON_EXE. Create/activate .venv first." >&2
    exit 1
  fi
  if command -v python3 >/dev/null 2>&1; then
  if [[ -n "$BATCH_FILE" ]]; then
    BATCH_PATH="$(resolve_repo_path "$BATCH_FILE")"
    if [[ ! -f "$BATCH_PATH" ]]; then
      echo "Batch file not found at $BATCH_PATH" >&2
      exit 1
    fi
    EXT="${BATCH_PATH##*.}"
    EXT_LOWER="$(echo "$EXT" | tr '[:upper:]' '[:lower:]')"
    if [[ "$EXT_LOWER" == "yaml" || "$EXT_LOWER" == "yml" || "$EXT_LOWER" == "json" ]]; then
      CONFIG_PATH="$BATCH_PATH"
    else
      TEMP_CONFIG_PATH="$(mktemp -t flouds_batch_XXXXXX.json)"
      "$PYTHON_EXE" - "$BATCH_PATH" batch "$TEMP_CONFIG_PATH" <<'PY'
    echo "Batch config not found at $CONFIG_PATH" >&2
    exit 1
  fi
fi

if [[ -n "$TEXT_FILE" ]]; then
  TEXT_FILE="$(resolve_repo_path "$TEXT_FILE")"
  if [[ ! -f "$TEXT_FILE" ]]; then
  if [[ -n "$BATCH_FILE" ]]; then
    BATCH_PATH="$(resolve_repo_path "$BATCH_FILE")"
    if [[ ! -f "$BATCH_PATH" ]]; then
      echo "Batch file not found at $BATCH_PATH" >&2
      exit 1
    fi
    EXT="${BATCH_PATH##*.}"
    EXT_LOWER="$(echo "$EXT" | tr '[:upper:]' '[:lower:]')"
    if [[ "$EXT_LOWER" == "yaml" || "$EXT_LOWER" == "yml" || "$EXT_LOWER" == "json" ]]; then
      CONFIG_PATH="$BATCH_PATH"
    else
      TEMP_CONFIG_PATH="$(mktemp -t flouds_batch_XXXXXX.json)"
      "$PYTHON_EXE" - "$BATCH_PATH" batch "$TEMP_CONFIG_PATH" <<'PY'
  import json
  import re
  import shlex
  import sys

  text_path = sys.argv[1]
  preset_name = sys.argv[2]
  out_path = sys.argv[3]

  value_map = {
      "--model-name": "model_name",
      "--model-for": "model_for",
      "--task": "task",
      "--model-folder": "model_folder",
      "--onnx-path": "onnx_path",
      "--framework": "framework",
      "--opset-version": "opset_version",
      "--device": "device",
      "--quantize": "quantize",
      "--pack-single-threshold-mb": "pack_single_threshold_mb",
      "--hf-token": "hf_token",
      "--library": "library",
  }
  flag_map = {
      "--optimize": "optimize",
      "--trust-remote-code": "trust_remote_code",
      "--normalize-embeddings": "normalize_embeddings",
      "--require-validator": "require_validator",
      "--skip-validator": "skip_validator",
      "--force": "force",
      "--pack-single-file": "pack_single_file",
      "--use-external-data-format": "use_external_data_format",
      "--no-local-prep": "no_local_pre",
      "--merge": "merge",
      "--cleanup": "cleanup",
      "--prune-canonical": "prune_canonical",
      "--no-post-process": "no_post_process",
      "--portable": "portable",
      "--use-sub-process": "use_subprocess",
      "--low-memory-env": "low_memory_env",
  }

  entries = []
  with open(text_path, "r", encoding="utf-8") as fh:
      for raw in fh:
          line = raw.strip()
          if not line or line.startswith("#"):
              continue
          if line.endswith(","):
              line = line[:-1].rstrip()
          if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
              line = line[1:-1].strip()

          if not re.match(r"^flouds-export\s+export(?:\s+|$)", line):
            continue

          tokens = shlex.split(line)
          if len(tokens) < 2:
              continue

          tokens = tokens[2:]

          entry = {}
          i = 0
          while i < len(tokens):
              tok = tokens[i]
              if tok in value_map:
                  if i + 1 < len(tokens):
                      key = value_map[tok]
                      val = tokens[i + 1]
                      if key in {"opset_version", "pack_single_threshold_mb"}:
                          try:
                              val = int(val)
                          except ValueError:
                              pass
                      entry[key] = val
                      i += 2
                      continue
              if tok in flag_map:
                  entry[flag_map[tok]] = True
              i += 1

          if "model_name" not in entry:
              continue
          entry.setdefault("model_for", "fe")
          entry.setdefault("task", "feature-extraction")
          entries.append(entry)

  if not entries:
      raise SystemExit(f"No valid export lines were found in text file: {text_path}")

  cfg = {"batch_presets": {preset_name: entries}}
  with open(out_path, "w", encoding="utf-8") as out:
      json.dump(cfg, out, indent=2)
  PY
      CONFIG_PATH="$TEMP_CONFIG_PATH"
    fi
  fi

import sys

text_path = sys.argv[1]
preset_name = sys.argv[2]
out_path = sys.argv[3]

value_map = {
    "--model-name": "model_name",
    "--model-for": "model_for",
    "--task": "task",
    "--model-folder": "model_folder",
    "--onnx-path": "onnx_path",
    "--framework": "framework",
    "--opset-version": "opset_version",
    "--device": "device",
    "--quantize": "quantize",
    "--pack-single-threshold-mb": "pack_single_threshold_mb",
    "--hf-token": "hf_token",
    "--library": "library",
}
flag_map = {
    "--optimize": "optimize",
    "--trust-remote-code": "trust_remote_code",
    "--normalize-embeddings": "normalize_embeddings",
    "--require-validator": "require_validator",
    "--skip-validator": "skip_validator",
    "--force": "force",
    "--pack-single-file": "pack_single_file",
    "--use-external-data-format": "use_external_data_format",
    "--no-local-prep": "no_local_prep",
    "--merge": "merge",
    "--cleanup": "cleanup",
    "--prune-canonical": "prune_canonical",
    "--no-post-process": "no_post_process",
    "--portable": "portable",
    "--use-sub-process": "use_subprocess",
    "--low-memory-env": "low_memory_env",
}

entries = []
with open(text_path, "r", encoding="utf-8") as fh:
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(","):
            line = line[:-1].rstrip()
        if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            line = line[1:-1].strip()

        if not re.match(r"^flouds-export\s+export(?:\s+|$)", line):
          continue

        tokens = shlex.split(line)
        if len(tokens) < 2:
            continue

        tokens = tokens[2:]

        entry = {}
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in value_map:
                if i + 1 < len(tokens):
                    key = value_map[tok]
                    val = tokens[i + 1]
                    if key in {"opset_version", "pack_single_threshold_mb"}:
                        try:
                            val = int(val)
                        except ValueError:
                            pass
                    entry[key] = val
                    i += 2
                    continue
            if tok in flag_map:
                entry[flag_map[tok]] = True
            i += 1

        if "model_name" not in entry:
            continue
        entry.setdefault("model_for", "fe")
        entry.setdefault("task", "feature-extraction")
        entries.append(entry)

if not entries:
    raise SystemExit(f"No valid export lines were found in text file: {text_path}")

cfg = {"batch_presets": {preset_name: entries}}
with open(out_path, "w", encoding="utf-8") as out:
    json.dump(cfg, out, indent=2)
PY

  CONFIG_PATH="$TEMP_CONFIG_PATH"
fi

CLI_ARGS=("-m" "model_exporter.cli.main" "batch" "--preset" "$PRESET" "--min-free-memory-gb" "$MIN_FREE_MEMORY_GB")

if [[ -n "$CONFIG_PATH" ]]; then
  CLI_ARGS+=("--config" "$CONFIG_PATH")
fi

if [[ "$FORCE" -eq 1 ]]; then
  CLI_ARGS+=("--force")
fi
if [[ "$SKIP_VALIDATOR" -eq 1 ]]; then
  CLI_ARGS+=("--skip-validator")
fi
if [[ "$OPTIMIZE" -eq 1 ]]; then
  CLI_ARGS+=("--optimize")
fi
if [[ "$CLEANUP" -eq 1 ]]; then
  CLI_ARGS+=("--cleanup")
fi
if [[ "$PRUNE_CANONICAL" -eq 1 ]]; then
  CLI_ARGS+=("--prune-canonical")
fi
if [[ "$NO_LOCAL_PREP" -eq 1 ]]; then
  CLI_ARGS+=("--no-local-prep")
fi
if [[ "$PORTABLE" -eq 1 ]]; then
  CLI_ARGS+=("--portable")
fi
if [[ "$FAIL_FAST" -eq 1 ]]; then
  CLI_ARGS+=("--fail-fast")
fi

if [[ "$LOG_TO_FILE" -eq 1 ]]; then
  CLI_ARGS+=("--log-to-file")
fi

printf 'Running wrapper command: %s %s\n' "$PYTHON_EXE" "${CLI_ARGS[*]}"

EXIT_CODE=0
set +e
"$PYTHON_EXE" "${CLI_ARGS[@]}"
EXIT_CODE=$?
set -e

if [[ -n "$TEMP_CONFIG_PATH" && -f "$TEMP_CONFIG_PATH" ]]; then
  rm -f "$TEMP_CONFIG_PATH" || true
fi

exit "$EXIT_CODE"
