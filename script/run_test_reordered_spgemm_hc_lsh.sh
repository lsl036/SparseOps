#!/bin/bash
# Run test_reordered_spgemm_hc_lsh for each dataset in testdatasets.txt.
# Uses reordering files: Graph Partition (gp_order) and Hypergraph Partition (hp_order).
# Default kernel=3. Command: test_reordered_spgemm_hc_lsh A.mtx B.mtx reordered_A --kernel=3
#
# Usage: from project root: bash script/run_test_reordered_spgemm_hc_lsh.sh
#        from build:        bash ../script/run_test_reordered_spgemm_hc_lsh.sh
# Env:   BUILD_DIR, KERNEL (default 3), BASE_DIR (default /data/suitesparse_collection),
#        GP_ORDER_DIR (default .../gp_order), HP_ORDER_DIR (default .../hp_order)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
DATASETS="${DATASETS_FILE:-$ROOT/testdatasets.txt}"
KERNEL="${KERNEL:-3}"
BASE="${BASE_DIR:-/data/suitesparse_collection}"
GP_DIR="${GP_ORDER_DIR:-/data2/linshengle_data/SpGEMM-Reordering/gp_order}"
HP_DIR="${HP_ORDER_DIR:-/data2/linshengle_data/SpGEMM-Reordering/hp_order}"

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
echo "Graph Partition reorder: $GP_DIR"
echo "Hypergraph Partition reorder: $HP_DIR"
echo "========================================"

run_one() {
  local name="$1"
  local reorder_label="$2"
  local reorder_file="$3"
  local A="$BASE/$name/$name.mtx"
  local B="$BASE/$name/$name.mtx"

  if [[ ! -f "$A" ]]; then
    echo "Skip $name ($reorder_label): A not found $A"
    return 0
  fi
  if [[ ! -f "$reorder_file" ]]; then
    echo "Skip $name ($reorder_label): reorder file not found $reorder_file"
    return 0
  fi

  echo ""
  echo "===== Dataset: $name ($reorder_label) ====="
  ./test_reordered_spgemm_hc_lsh "$A" "$B" "$reorder_file" --kernel="$KERNEL"
}

while IFS= read -r name || [[ -n "$name" ]]; do
  name=$(echo "$name" | tr -d '\r')
  [[ -z "$name" ]] && continue

  run_one "$name" "Graph Partition" "$GP_DIR/$name.gporder"
  run_one "$name" "Hypergraph Partition" "$HP_DIR/$name.hporder"
done < "$DATASETS"

echo ""
echo "========================================"
echo "Done."
