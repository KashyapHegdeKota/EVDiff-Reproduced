#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

EVDIFF_ROOT="${EVDIFF_ROOT:-$WORKSPACE_ROOT/EvDiff}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RAW_ROOT="${RAW_ROOT:-$WORKSPACE_ROOT/data/DSEC/train}"
CONVERTED_ROOT="${CONVERTED_ROOT:-$WORKSPACE_ROOT/data/DSEC_mem/train}"
PREVIEW_ROOT="${PREVIEW_ROOT:-$WORKSPACE_ROOT/conversion_preview}"
LOG_FILE="${LOG_FILE:-$WORKSPACE_ROOT/dsec_preparation.log}"

SMOKE_SEQUENCE="zurich_city_02_a"

ALL_SEQUENCES=(
  "zurich_city_00_a"
  "zurich_city_02_a"
  "zurich_city_04_b"
)

REMAINING_SEQUENCES=(
  "zurich_city_00_a"
  "zurich_city_04_b"
)

FORCE_CONVERSION=false


usage() {
  cat <<'EOF'
Usage:
  ./run_dsec_preparation.sh [options]

Options:
  --force       Convert sequences again even if outputs already exist
  --help        Display this help message

Environment variables:
  EVDIFF_ROOT       Official EvDiff repository directory
  PYTHON_BIN        Python executable, default: python
  RAW_ROOT          Raw DSEC dataset directory
  CONVERTED_ROOT    Converted DSEC dataset directory
  PREVIEW_ROOT      RGB preview directory
  LOG_FILE          Preparation log path
EOF
}


while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE_CONVERSION=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done


log_section() {
  echo
  echo "========================================================================"
  echo "$1"
  echo "========================================================================"
}


fail() {
  echo
  echo "[FAILED] $1" >&2
  exit 1
}


on_error() {
  local exit_code=$?
  local line_number=$1

  echo
  echo "[FAILED] DSEC preparation failed on line $line_number."
  echo "Review the log file:"
  echo "  $LOG_FILE"

  exit "$exit_code"
}


trap 'on_error $LINENO' ERR


mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1


log_section "EVDIFF DSEC PREPARATION"

echo "Workspace root: $WORKSPACE_ROOT"
echo "Utilities:      $SCRIPT_DIR"
echo "EvDiff root:    $EVDIFF_ROOT"
echo "Python:         $PYTHON_BIN"
echo "Raw dataset:    $RAW_ROOT"
echo "Converted data: $CONVERTED_ROOT"
echo "RGB previews:   $PREVIEW_ROOT"
echo "Log file:       $LOG_FILE"
echo "Force convert:  $FORCE_CONVERSION"


log_section "CHECKING DIRECTORIES"

if [[ ! -d "$EVDIFF_ROOT" ]]; then
  fail "EvDiff repository not found: $EVDIFF_ROOT"
fi

if [[ ! -d "$RAW_ROOT" ]]; then
  fail "Raw DSEC dataset not found: $RAW_ROOT"
fi

echo "[OK] EvDiff repository: $EVDIFF_ROOT"
echo "[OK] Raw DSEC dataset: $RAW_ROOT"


log_section "CHECKING REQUIRED SCRIPTS"

REQUIRED_SCRIPTS=(
  "verify_raw_dsec.py"
  "convert_dsec.py"
  "inspect_dsec_arrays.py"
  "export_dsec_previews.py"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
  script_path="$SCRIPT_DIR/$script"

  if [[ ! -f "$script_path" ]]; then
    fail "Missing required script: $script_path"
  fi

  echo "[OK] $script_path"
done

CONVERTER_PATH="$EVDIFF_ROOT/tools/dsec/convert_small_align_rgb.py"

if [[ ! -f "$CONVERTER_PATH" ]]; then
  fail "Missing EvDiff RGB converter: $CONVERTER_PATH"
fi

echo "[OK] $CONVERTER_PATH"


log_section "CHECKING PYTHON ENVIRONMENT"

cd "$EVDIFF_ROOT"

export PYTHONPATH="$EVDIFF_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - <<'PY'
import sys

print("Python:", sys.version)
print("Executable:", sys.executable)

try:
    import numpy
    print("NumPy:", numpy.__version__)
except ImportError as error:
    raise SystemExit(f"NumPy is unavailable: {error}")

try:
    import cv2
    print("OpenCV:", cv2.__version__)
except ImportError as error:
    raise SystemExit(f"OpenCV is unavailable: {error}")

try:
    from tools.dsec.convert_small_align_rgb import DSECToHQFConverter
    print("EvDiff converter import: successful")
except ImportError as error:
    raise SystemExit(f"EvDiff converter import failed: {error}")
PY


mkdir -p "$CONVERTED_ROOT"
mkdir -p "$PREVIEW_ROOT"


log_section "STEP 1: VERIFYING RAW DSEC DATA"

"$PYTHON_BIN" "$SCRIPT_DIR/verify_raw_dsec.py" \
  --raw-root "$RAW_ROOT" \
  --sequences "${ALL_SEQUENCES[@]}"


log_section "STEP 2: CONVERTING SMOKE-TEST SEQUENCE"

CONVERT_ARGS=(
  --raw-root "$RAW_ROOT"
  --output-root "$CONVERTED_ROOT"
  --sequences "$SMOKE_SEQUENCE"
)

if [[ "$FORCE_CONVERSION" == true ]]; then
  CONVERT_ARGS+=(--force)
fi

"$PYTHON_BIN" "$SCRIPT_DIR/convert_dsec.py" \
  "${CONVERT_ARGS[@]}"


log_section "STEP 3: INSPECTING SMOKE-TEST ARRAYS"

"$PYTHON_BIN" "$SCRIPT_DIR/inspect_dsec_arrays.py" \
  --input-root "$CONVERTED_ROOT" \
  --sequences "$SMOKE_SEQUENCE"


log_section "STEP 4: EXPORTING SMOKE-TEST RGB PREVIEWS"

"$PYTHON_BIN" "$SCRIPT_DIR/export_dsec_previews.py" \
  --input-root "$CONVERTED_ROOT" \
  --output-root "$PREVIEW_ROOT" \
  --sequences "$SMOKE_SEQUENCE" \
  --count 5


log_section "STEP 5: CONVERTING REMAINING SEQUENCES"

CONVERT_ARGS=(
  --raw-root "$RAW_ROOT"
  --output-root "$CONVERTED_ROOT"
  --sequences "${REMAINING_SEQUENCES[@]}"
)

if [[ "$FORCE_CONVERSION" == true ]]; then
  CONVERT_ARGS+=(--force)
fi

"$PYTHON_BIN" "$SCRIPT_DIR/convert_dsec.py" \
  "${CONVERT_ARGS[@]}"


log_section "STEP 6: INSPECTING ALL CONVERTED ARRAYS"

"$PYTHON_BIN" "$SCRIPT_DIR/inspect_dsec_arrays.py" \
  --input-root "$CONVERTED_ROOT" \
  --sequences "${ALL_SEQUENCES[@]}"


log_section "STEP 7: EXPORTING ALL RGB PREVIEWS"

"$PYTHON_BIN" "$SCRIPT_DIR/export_dsec_previews.py" \
  --input-root "$CONVERTED_ROOT" \
  --output-root "$PREVIEW_ROOT" \
  --sequences "${ALL_SEQUENCES[@]}" \
  --count 5


log_section "DSEC PREPARATION COMPLETED"

echo "Successfully prepared:"

for sequence in "${ALL_SEQUENCES[@]}"; do
  echo "  - $sequence"
done

echo
echo "Converted dataset:"
echo "  $CONVERTED_ROOT"

echo
echo "RGB previews:"
echo "  $PREVIEW_ROOT"

echo
echo "Complete log:"
echo "  $LOG_FILE"

echo
echo "Inspect the generated preview frames before inference."