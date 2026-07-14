#!/bin/bash
# Run test_reordered_spgemm_hc_lsh for each dataset in testdatasets.txt.
# By default, matrices and reorderings live under SPARSEOPS_ROOT/data/.
# Default kernel=3. Command: test_reordered_spgemm_hc_lsh A.mtx B.mtx reordered_A --kernel=3
#
# Usage:
#   bash script/run_test_reordered_spgemm_hc_lsh.sh
#   bash script/run_test_reordered_spgemm_hc_lsh.sh rcm gray
#   REORDER_TYPES=gp,hp bash script/run_test_reordered_spgemm_hc_lsh.sh
#
# Env:
#   SPARSEOPS_ROOT (default: inferred from this script),
#   BUILD_DIR (default SPARSEOPS_ROOT/build),
#   DATASETS_FILE (default SPARSEOPS_ROOT/testdatasets.txt),
#   BASE_DIR (default SPARSEOPS_ROOT/data),
#   REORDER_BASE_DIR (default SPARSEOPS_ROOT/data/reordering),
#   KERNEL (default 3),
#   REORDER_TYPES (comma-separated: gp,hp,rcm,gray; default gp,hp),
#   GP_ORDER_DIR / HP_ORDER_DIR / RCM_ORDER_DIR / GRAY_ORDER_DIR (optional overrides),
#   RESULT_CSV (default CALLER_DIR/reordered_spgemm_hc_lsh_results.csv),
#   THREADS (optional): passes --threads=<n> to the binary

set -e
CALLER_DIR="$(pwd -P)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPARSEOPS_ROOT="${SPARSEOPS_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
BUILD="${BUILD_DIR:-${SPARSEOPS_ROOT}/build}"
DATASETS="${DATASETS_FILE:-${SPARSEOPS_ROOT}/testdatasets.txt}"
KERNEL="${KERNEL:-3}"
THREADS="${THREADS:-}"
BASE="${BASE_DIR:-${SPARSEOPS_ROOT}/data}"
REORDER_BASE="${REORDER_BASE_DIR:-${SPARSEOPS_ROOT}/data/reordering}"
RESULT_CSV="${RESULT_CSV:-$CALLER_DIR/reordered_spgemm_hc_lsh_results.csv}"

export SPARSEOPS_ROOT

GP_DIR="${GP_ORDER_DIR:-$REORDER_BASE/gp_order}"
HP_DIR="${HP_ORDER_DIR:-$REORDER_BASE/hp_order}"
RCM_DIR="${RCM_ORDER_DIR:-$REORDER_BASE/rcm_order}"
GRAY_DIR="${GRAY_ORDER_DIR:-$REORDER_BASE/gray_order}"

SELECTED_ORDERS=()

parse_selected_orders() {
  if [[ -n "${REORDER_TYPES:-}" ]]; then
    local part
    IFS=',' read -ra SELECTED_ORDERS <<< "${REORDER_TYPES}"
    for part in "${SELECTED_ORDERS[@]}"; do
      case "$part" in
        gp|hp|rcm|gray) ;;
        *)
          echo "Error: unknown order type in REORDER_TYPES: $part (use gp,hp,rcm,gray)"
          exit 1
          ;;
      esac
    done
    return
  fi

  local arg
  for arg in "$@"; do
    case "$arg" in
      gp|hp|rcm|gray) SELECTED_ORDERS+=("$arg") ;;
      *)
        echo "Error: unknown argument: $arg (use gp, hp, rcm, gray)"
        exit 1
        ;;
    esac
  done

  if [[ ${#SELECTED_ORDERS[@]} -eq 0 ]]; then
    SELECTED_ORDERS=(gp hp)
  fi
}

reorder_file_for() {
  local order="$1"
  local name="$2"
  case "$order" in
    gp)   echo "$GP_DIR/${name}.gporder" ;;
    hp)   echo "$HP_DIR/${name}.hporder" ;;
    rcm)  echo "$RCM_DIR/${name}.rcmorder" ;;
    gray) echo "$GRAY_DIR/${name}.grayorder" ;;
    *)
      echo "Error: unknown order type: $order" >&2
      return 1
      ;;
  esac
}

reorder_label_for() {
  local order="$1"
  case "$order" in
    gp)   echo "Graph Partition" ;;
    hp)   echo "Hypergraph Partition" ;;
    rcm)  echo "RCM" ;;
    gray) echo "Gray" ;;
    *)    echo "$order" ;;
  esac
}

parse_selected_orders "$@"

if [[ ! -f "$DATASETS" ]]; then
  echo "Error: dataset list not found at $DATASETS"
  exit 1
fi
if [[ ! -x "$BUILD/test_reordered_spgemm_hc_lsh" ]]; then
  echo "Error: test_reordered_spgemm_hc_lsh not found at $BUILD/test_reordered_spgemm_hc_lsh (build first)"
  exit 1
fi

cd "$BUILD"
echo "SparseOps root: $SPARSEOPS_ROOT"
echo "Using build: $BUILD, kernel: $KERNEL"
echo "Matrix base: $BASE"
if [[ -n "$THREADS" ]]; then
  echo "Threads: $THREADS (passed as --threads=$THREADS)"
else
  echo "Threads: (default from binary / OpenMP)"
fi
echo "Reorder base: $REORDER_BASE"
echo "Selected orders: ${SELECTED_ORDERS[*]}"
echo "Results CSV: $RESULT_CSV"
echo "========================================"

echo "mtx_name,kernel,reorder,Average_time_ms,Average_GFLOPS" > "$RESULT_CSV"

run_one() {
  local name="$1"
  local order_tag="$2"
  local reorder_file="$3"
  local reorder_label
  reorder_label="$(reorder_label_for "$order_tag")"

  local A="$BASE/$name/$name.mtx"
  local B="$BASE/$name/$name.mtx"

  if [[ ! -f "$A" ]]; then
    echo "Skip $name ($reorder_label): A not found $A"
    echo "${name},${KERNEL},${order_tag},," >> "$RESULT_CSV"
    return 0
  fi
  if [[ ! -f "$reorder_file" ]]; then
    echo "Skip $name ($reorder_label): reorder file not found $reorder_file"
    echo "${name},${KERNEL},${order_tag},," >> "$RESULT_CSV"
    return 0
  fi

  echo ""
  echo "===== Dataset: $name ($reorder_label) ====="

  local out rc=0
  set +e
  if [[ -n "$THREADS" ]]; then
    out=$(./test_reordered_spgemm_hc_lsh "$A" "$B" "$reorder_file" --kernel="$KERNEL" --threads="$THREADS" 2>&1)
  else
    out=$(./test_reordered_spgemm_hc_lsh "$A" "$B" "$reorder_file" --kernel="$KERNEL" 2>&1)
  fi
  rc=$?
  set -e

  printf '%s\n' "$out"

  local avg_ms gflops
  avg_ms=$(printf '%s\n' "$out" | sed -n 's/.*Average time (LeSpGEMM_VLength): \([0-9.]*\) ms.*/\1/p' | head -n1)
  gflops=$(printf '%s\n' "$out" | sed -n 's/.*Average GFLOPS: \([0-9.]*\).*/\1/p' | head -n1)

  if [[ $rc -ne 0 ]] || [[ -z "$avg_ms" ]] || [[ -z "$gflops" ]]; then
    echo "${name},${KERNEL},${order_tag},," >> "$RESULT_CSV"
    echo "  WARN: exit code $rc (empty metrics = skip or parse failed)" >&2
  else
    echo "${name},${KERNEL},${order_tag},${avg_ms},${gflops}" >> "$RESULT_CSV"
  fi
}

while IFS= read -r name || [[ -n "$name" ]]; do
  name=$(echo "$name" | tr -d '\r')
  [[ -z "$name" ]] && continue

  local_order=""
  for local_order in "${SELECTED_ORDERS[@]}"; do
    run_one "$name" "$local_order" "$(reorder_file_for "$local_order" "$name")"
  done
done < "$DATASETS"

echo ""
echo "========================================"
echo "Done. Wrote $RESULT_CSV"
