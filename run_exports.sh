#!/usr/bin/env bash
# Optional Linux/macOS convenience wrapper for batch exports.
# Canonical batch entrypoint: flouds-export batch --preset recommended
#
# Usage:
#   ./run_exports.sh
#   ./run_exports.sh --use-venv --force
#   ./run_exports.sh --config ./docs/batch_presets_example.yaml --preset text-import
#   ./run_exports.sh --text-file ./docs/batch_commands.txt --preset text-import
#   ./run_exports.sh --batch-file ./docs/batch_presets_full.yaml --preset full
#
# Flags:
#   --use-venv              Require the repository .venv Python interpreter
#   --force                 Apply --force to all batch exports
#   --optimize              Apply --optimize to all batch exports
#   --optimization-level    Apply --optimization-level to all batch exports; requires --optimize
#   --cleanup               Apply --cleanup to all batch exports
#   --skip-validator        Apply --skip-validator to all batch exports unless active entries require validation
#   --prune-canonical       Apply --prune-canonical to all batch exports; requires --cleanup
#   --portable              Apply --portable to all batch exports; requires --optimize
#   --fail-fast             Stop batch on first failed export
#   --config PATH           YAML/JSON batch config file
#   --text-file PATH        Text file containing one `flouds-export export ...` command per line
#   --batch-file PATH       YAML/JSON config or text command file, detected by extension
#   --preset NAME           Preset name to run
#   --min-free-memory-gb N  Minimum free memory in GB before each export (default: 1)
#   --log-to-file           Request per-export log files

set -euo pipefail

USE_VENV=0
FORCE=0
SKIP_VALIDATOR=0
OPTIMIZE=0
CLEANUP=0
PRUNE_CANONICAL=0
PORTABLE=0
FAIL_FAST=0
LOG_TO_FILE=0
CONFIG_PATH=""
TEXT_FILE=""
BATCH_FILE=""
PRESET="recommended"
PRESET_SET=0
OPTIMIZATION_LEVEL=""
MIN_FREE_MEMORY_GB=1
TEMP_CONFIG_PATH=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

usage() {
  sed -n '1,44p' "$0" | sed 's/^# \{0,1\}//'
}

resolve_repo_path() {
  local input_path="$1"
  if [[ -z "$input_path" ]]; then
    printf '%s' ""
  elif [[ "$input_path" = /* ]]; then
    printf '%s' "$input_path"
  else
    printf '%s' "$REPO_ROOT/$input_path"
  fi
}

infer_first_preset() {
  local config_path="$1"
  local ext="${config_path##*.}"
  local ext_lower
  ext_lower="$(printf '%s' "$ext" | tr '[:upper:]' '[:lower:]')"

  if [[ "$ext_lower" == "json" ]]; then
    "$PYTHON_EXE" - "$config_path" <<'PY' || true
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
presets = data.get("batch_presets") or {}
if isinstance(presets, dict) and presets:
    print(next(iter(presets)))
PY
    return
  fi

  awk '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*batch_presets[[:space:]]*:[[:space:]]*$/ { inside=1; next }
    inside && /^  [A-Za-z0-9_.-][A-Za-z0-9_.-]*[[:space:]]*:[[:space:]]*$/ {
      line=$0
      sub(/^  /, "", line)
      sub(/[[:space:]]*:[[:space:]]*$/, "", line)
      print line
      exit
    }
  ' "$config_path"
}

convert_text_file_to_config() {
  local text_path="$1"
  local preset_name="$2"
  local out_path="$3"

  "$PYTHON_EXE" - "$text_path" "$preset_name" "$out_path" <<'PY'
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
    "--optimization-level": "optimization_level",
    "--opset-version": "opset_version",
    "--device": "device",
    "--quantize": "quantize",
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
    "--log-to-file": "log_to_file",
}

integer_keys = {"optimization_level", "opset_version"}
entries = []

with open(text_path, "r", encoding="utf-8") as handle:
    for line_number, raw in enumerate(handle, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(","):
            line = line[:-1].rstrip()
        if (line.startswith('"') and line.endswith('"')) or (
            line.startswith("'") and line.endswith("'")
        ):
            line = line[1:-1].strip()

        if not re.match(r"^flouds-export\s+export(?:\s+|$)", line):
            continue

        tokens = shlex.split(line)
        if len(tokens) < 2:
            continue

        entry = {}
        i = 2
        while i < len(tokens):
            token = tokens[i]
            if token in value_map:
                if i + 1 >= len(tokens):
                    raise SystemExit(f"{text_path}:{line_number}: missing value for {token}")
                key = value_map[token]
                value = tokens[i + 1]
                if key in integer_keys:
                    value = int(value)
                entry[key] = value
                i += 2
                continue
            if token in flag_map:
                entry[flag_map[token]] = True
                i += 1
                continue
            if token.startswith("--"):
                raise SystemExit(f"{text_path}:{line_number}: unsupported export flag {token}")
            i += 1

        if "model_name" not in entry:
            continue
        entry.setdefault("model_for", "fe")
        entry.setdefault("task", "feature-extraction")
        entries.append(entry)

if not entries:
    raise SystemExit(f"No valid export lines were found in text file: {text_path}")

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump({"batch_presets": {preset_name: entries}}, handle, indent=2)
PY
}

cleanup() {
  if [[ -n "$TEMP_CONFIG_PATH" && -f "$TEMP_CONFIG_PATH" ]]; then
    rm -f "$TEMP_CONFIG_PATH" || true
  fi
}
trap cleanup EXIT

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
    --optimization-level|-OptimizationLevel)
      OPTIMIZATION_LEVEL="${2:-}"
      shift 2
      ;;
    --cleanup|-Cleanup)
      CLEANUP=1
      shift
      ;;
    --prune-canonical|-PruneCanonical)
      PRUNE_CANONICAL=1
      shift
      ;;
    --portable|-Portable)
      PORTABLE=1
      shift
      ;;
    --no-local-prep|-NoLocalPrep)
      echo "--no-local-prep is only valid for LLM exports. Set it per LLM entry in config/text files." >&2
      exit 1
      ;;
    --fail-fast|-FailFast)
      FAIL_FAST=1
      shift
      ;;
    --config|-Config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --text-file|-TextFile)
      TEXT_FILE="${2:-}"
      shift 2
      ;;
    --batch-file|-BatchFile)
      BATCH_FILE="${2:-}"
      shift 2
      ;;
    --preset|-Preset)
      PRESET="${2:-}"
      PRESET_SET=1
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

VENV_PYTHON_EXE="$REPO_ROOT/.venv/bin/python"
if [[ "$USE_VENV" -eq 1 || -x "$VENV_PYTHON_EXE" ]]; then
  if [[ ! -x "$VENV_PYTHON_EXE" ]]; then
    echo "Python venv not found at $VENV_PYTHON_EXE. Create/activate .venv first." >&2
    exit 1
  fi
  PYTHON_EXE="$VENV_PYTHON_EXE"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_EXE="python3"
else
  PYTHON_EXE="python"
fi

if [[ -d "$REPO_ROOT/src" ]]; then
  if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$REPO_ROOT/src"
  else
    export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"
  fi
fi

case "$OPTIMIZATION_LEVEL" in
  ""|0|1|2|99) ;;
  *)
    echo "--optimization-level must be one of: 0, 1, 2, 99" >&2
    exit 1
    ;;
esac

if [[ "$PORTABLE" -eq 1 && "$OPTIMIZE" -ne 1 ]]; then
  echo "--portable requires --optimize." >&2
  exit 1
fi
if [[ -n "$OPTIMIZATION_LEVEL" && "$OPTIMIZE" -ne 1 ]]; then
  echo "--optimization-level requires --optimize." >&2
  exit 1
fi
if [[ "$PRUNE_CANONICAL" -eq 1 && "$CLEANUP" -ne 1 ]]; then
  echo "--prune-canonical requires --cleanup." >&2
  exit 1
fi
if [[ -n "$CONFIG_PATH" && -n "$TEXT_FILE" ]]; then
  echo "Use only one of --config or --text-file." >&2
  exit 1
fi
if [[ -n "$BATCH_FILE" && ( -n "$CONFIG_PATH" || -n "$TEXT_FILE" ) ]]; then
  echo "Use --batch-file by itself, or use --config/--text-file explicitly." >&2
  exit 1
fi

if [[ -n "$BATCH_FILE" ]]; then
  BATCH_PATH="$(resolve_repo_path "$BATCH_FILE")"
  if [[ ! -f "$BATCH_PATH" ]]; then
    echo "Batch file not found at $BATCH_PATH" >&2
    exit 1
  fi
  EXT="${BATCH_PATH##*.}"
  EXT_LOWER="$(printf '%s' "$EXT" | tr '[:upper:]' '[:lower:]')"
  if [[ "$EXT_LOWER" == "yaml" || "$EXT_LOWER" == "yml" || "$EXT_LOWER" == "json" ]]; then
    CONFIG_PATH="$BATCH_PATH"
  else
    TEXT_FILE="$BATCH_PATH"
  fi
fi

CLI_ARGS=("-m" "model_exporter.cli.main" "batch" "--min-free-memory-gb" "$MIN_FREE_MEMORY_GB")

if [[ -n "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$(resolve_repo_path "$CONFIG_PATH")"
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Batch config not found at $CONFIG_PATH" >&2
    exit 1
  fi
  if [[ "$PRESET_SET" -eq 0 ]]; then
    INFERRED_PRESET="$(infer_first_preset "$CONFIG_PATH")"
    if [[ -n "$INFERRED_PRESET" ]]; then
      PRESET="$INFERRED_PRESET"
    fi
  fi
  CLI_ARGS+=("--config" "$CONFIG_PATH")
fi

if [[ -n "$TEXT_FILE" ]]; then
  TEXT_FILE="$(resolve_repo_path "$TEXT_FILE")"
  if [[ ! -f "$TEXT_FILE" ]]; then
    echo "Text file not found at $TEXT_FILE" >&2
    exit 1
  fi
  if [[ "$PRESET_SET" -eq 0 && "$PRESET" == "recommended" ]]; then
    PRESET="batch"
  fi
  TEMP_CONFIG_PATH="$(mktemp "${TMPDIR:-/tmp}/flouds_batch_XXXXXX.json")"
  convert_text_file_to_config "$TEXT_FILE" "$PRESET" "$TEMP_CONFIG_PATH"
  CLI_ARGS+=("--config" "$TEMP_CONFIG_PATH")
fi

CLI_ARGS+=("--preset" "$PRESET")

if [[ "$FORCE" -eq 1 ]]; then
  CLI_ARGS+=("--force")
fi
if [[ "$SKIP_VALIDATOR" -eq 1 ]]; then
  CLI_ARGS+=("--skip-validator")
fi
if [[ "$OPTIMIZE" -eq 1 ]]; then
  CLI_ARGS+=("--optimize")
fi
if [[ -n "$OPTIMIZATION_LEVEL" ]]; then
  CLI_ARGS+=("--optimization-level" "$OPTIMIZATION_LEVEL")
fi
if [[ "$CLEANUP" -eq 1 ]]; then
  CLI_ARGS+=("--cleanup")
fi
if [[ "$PRUNE_CANONICAL" -eq 1 ]]; then
  CLI_ARGS+=("--prune-canonical")
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
"$PYTHON_EXE" "${CLI_ARGS[@]}"
