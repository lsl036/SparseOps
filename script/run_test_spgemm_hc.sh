#!/bin/bash
# Run test_spgemm_hc for each dataset in testdatasets.txt.
# Usage: from project root, run: bash script/run_test_spgemm_hc.sh
#        or from build: bash ../script/run_test_spgemm_hc.sh
# Command pattern (like cant): test_spgemm_hc A.mtx B.mtx candidate_pairs.mtx --kernel=3

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
DATASETS="$ROOT/testdatasets.txt"
KERNEL="${KERNEL:-3}"

if [[ ! -f "$DATASETS" ]]; then
  echo "Error: testdatasets.txt not found at $DATASETS"
  exit 1
fi
if [[ ! -x "$BUILD/test_spgemm_hc" ]]; then
  echo "Error: test_spgemm_hc not found at $BUILD/test_spgemm_hc (build first)"
  exit 1
fi

cd "$BUILD"
echo "Using build dir: $BUILD, kernel: $KERNEL"
echo "========================================"

while IFS= read -r name || [[ -n "$name" ]]; do
  name=$(echo "$name" | tr -d '\r')
  [[ -z "$name" ]] && continue

  A="/data/suitesparse_collection/$name/$name.mtx"
  B="/data/suitesparse_collection/$name/$name.mtx"
  # AAt
  # CAND="/data2/linshengle_data/SpGEMM-Reordering/close_pairs/$name.mtx"
  # Graph Partition
  CAND="/data2/linshengle_data/SpGEMM-Reordering/gp_order/$name.mtx"
  # Hypergraph Partition
  # CAND="/data2/linshengle_data/SpGEMM-Reordering/hp_order/$name.mtx"
  
  if [[ ! -f "$A" ]]; then
    echo "Skip $name: A not found $A"
    continue
  fi
  if [[ ! -f "$CAND" ]]; then
    echo "Skip $name: candidate pairs not found $CAND"
    continue
  fi

  echo ""
  echo "===== Dataset: $name ====="
  ./test_spgemm_hc "$A" "$B" "$CAND" --kernel="$KERNEL"
done < "$DATASETS"

echo ""
echo "========================================"
echo "Done."

# 运行python分析脚本提取结果
# python3 parse_run_test_hc_output.py run_test_hc_AAt.out run_test_hc_AAt_results.csv