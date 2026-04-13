#!/bin/bash
# Run test_reordered_spgemm_hc_lsh for each dataset in testdatasets.txt.
# Uses reordering files: Graph Partition (gp_order) and Hypergraph Partition (hp_order).
# Default kernel=3. Command: test_reordered_spgemm_hc_lsh A.mtx B.mtx reordered_A --kernel=3
#
# Usage: from project root: bash script/run_test_reordered_spgemm_hc_lsh.sh
#        from build:        bash ../script/run_test_reordered_spgemm_hc_lsh.sh
# Env:   BUILD_DIR, KERNEL (default 3), BASE_DIR (default /data/suitesparse_collection),
#        GP_ORDER_DIR, HP_ORDER_DIR,
#        RESULT_CSV (default ROOT/reordered_spgemm_hc_lsh_results.csv): mtx_name,reorder,Average_time_ms,Average_GFLOPS
#        THREADS (optional): if set, passes --threads=<n> to the binary (e.g. THREADS=64)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
DATASETS="${DATASETS_FILE:-$ROOT/testdatasets.txt}"
KERNEL="${KERNEL:-3}"
THREADS="${THREADS:-}"
BASE="${BASE_DIR:-/data/suitesparse_collection}"
GP_DIR="${GP_ORDER_DIR:-/data/linshengle_data/SpGEMM-Reordering/gp_order}"
HP_DIR="${HP_ORDER_DIR:-/data/linshengle_data/SpGEMM-Reordering/hp_order}"
RESULT_CSV="${RESULT_CSV:-$ROOT/reordered_spgemm_hc_lsh_results.csv}"

if [[ ! -f "$DATASETS" ]]; then
  echo "Error: dataset list not found at $DATASETS"
  exit 1
fi
if [[ ! -x "$BUILD/test_reordered_spgemm_hc_lsh" ]]; then
  echo "Error: test_reordered_spgemm_hc_lsh not found at $BUILD/test_reordered_spgemm_hc_lsh (build first)"
  exit 1
fi

cd "$BUILD"
echo "Using build: $BUILD, kernel: $KERNEL"
if [[ -n "$THREADS" ]]; then
  echo "Threads: $THREADS (passed as --threads=$THREADS)"
else
  echo "Threads: (default from binary / OpenMP)"
fi
echo "Graph Partition reorder: $GP_DIR"
echo "Hypergraph Partition reorder: $HP_DIR"
echo "Results CSV: $RESULT_CSV"
echo "========================================"

echo "mtx_name,reorder,Average_time_ms,Average_GFLOPS" > "$RESULT_CSV"

run_one() {
  local name="$1"
  local order_tag="$2"
  local reorder_file="$3"
  local reorder_label
  case "$order_tag" in
    gp) reorder_label="Graph Partition" ;;
    hp) reorder_label="Hypergraph Partition" ;;
    *) reorder_label="$order_tag" ;;
  esac

  local A="$BASE/$name/$name.mtx"
  local B="$BASE/$name/$name.mtx"

  if [[ ! -f "$A" ]]; then
    echo "Skip $name ($reorder_label): A not found $A"
    echo "${name},${order_tag},," >> "$RESULT_CSV"
    return 0
  fi
  if [[ ! -f "$reorder_file" ]]; then
    echo "Skip $name ($reorder_label): reorder file not found $reorder_file"
    echo "${name},${order_tag},," >> "$RESULT_CSV"
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
    echo "${name},${order_tag},," >> "$RESULT_CSV"
    echo "  WARN: exit code $rc (empty metrics = skip or parse failed)" >&2
  else
    echo "${name},${order_tag},${avg_ms},${gflops}" >> "$RESULT_CSV"
  fi
}

while IFS= read -r name || [[ -n "$name" ]]; do
  name=$(echo "$name" | tr -d '\r')
  [[ -z "$name" ]] && continue

  run_one "$name" gp "$GP_DIR/$name.gporder"
  run_one "$name" hp "$HP_DIR/$name.hporder"
done < "$DATASETS"

echo ""
echo "========================================"
echo "Done. Wrote $RESULT_CSV"
