#!/bin/bash
# Run test_spgemm_hc for each dataset in a list file.
# Command pattern: test_spgemm_hc A.mtx B.mtx candidate_pairs.mtx [--threads=N] --kernel=K
#
# Usage: from project root: bash script/run_test_spgemm_hc.sh
#        from build:        bash ../script/run_test_spgemm_hc.sh
#
# Env:
#   BUILD_DIR, DATASETS_FILE (default ROOT/testdatasets.txt),
#   BASE_DIR (default /data/suitesparse_collection),
#   CLOSE_PAIRS_DIR (default .../close_pairs),
#   KERNEL (default 3), THREADS (optional, e.g. 64),
#   RESULT_CSV (default ROOT/test_spgemm_hc_results.csv)
# CSV columns: mtx_name,threads,kernel,Average_time_ms,Average_GFLOPS

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
DATASETS="${DATASETS_FILE:-$ROOT/testdatasets.txt}"
KERNEL="${KERNEL:-3}"
THREADS="${THREADS:-}"
BASE="${BASE_DIR:-/data/suitesparse_collection}"
CLOSE_PAIRS_DIR="${CLOSE_PAIRS_DIR:-/data/linshengle_data/SpGEMM-Reordering/close_pairs}"
RESULT_CSV="${RESULT_CSV:-$ROOT/test_spgemm_hc_results.csv}"

if [[ ! -f "$DATASETS" ]]; then
  echo "Error: dataset list not found at $DATASETS"
  exit 1
fi
if [[ ! -x "$BUILD/test_spgemm_hc" ]]; then
  echo "Error: test_spgemm_hc not found at $BUILD/test_spgemm_hc (build first)"
  exit 1
fi

cd "$BUILD"
echo "Using build: $BUILD"
echo "Kernel: $KERNEL"
if [[ -n "$THREADS" ]]; then
  echo "Threads: $THREADS (passed as --threads=$THREADS)"
else
  echo "Threads: (default from binary / OpenMP)"
fi
echo "BASE_DIR: $BASE"
echo "CLOSE_PAIRS_DIR: $CLOSE_PAIRS_DIR"
echo "Results CSV: $RESULT_CSV"
echo "========================================"

echo "mtx_name,threads,kernel,Average_time_ms,Average_GFLOPS" > "$RESULT_CSV"

while IFS= read -r name || [[ -n "$name" ]]; do
  name=$(echo "$name" | tr -d '\r')
  [[ -z "$name" ]] && continue

  A="$BASE/$name/$name.mtx"
  B="$BASE/$name/$name.mtx"
  CAND="$CLOSE_PAIRS_DIR/$name.mtx"

  if [[ ! -f "$A" ]]; then
    echo "Skip $name: A not found $A"
    echo "${name},${THREADS},${KERNEL},," >> "$RESULT_CSV"
    continue
  fi
  if [[ ! -f "$CAND" ]]; then
    echo "Skip $name: candidate pairs not found $CAND"
    echo "${name},${THREADS},${KERNEL},," >> "$RESULT_CSV"
    continue
  fi

  echo ""
  echo "===== Dataset: $name ====="

  thr="${THREADS:-}"
  out=""
  rc=0
  set +e
  if [[ -n "$thr" ]]; then
    out=$(./test_spgemm_hc "$A" "$B" "$CAND" --threads="$thr" --kernel="$KERNEL" 2>&1)
  else
    out=$(./test_spgemm_hc "$A" "$B" "$CAND" --kernel="$KERNEL" 2>&1)
  fi
  rc=$?
  set -e

  printf '%s\n' "$out"

  avg_ms=$(printf '%s\n' "$out" | sed -n 's/.*Average time: \([0-9.]*\) ms.*/\1/p' | head -n1)
  gflops=$(printf '%s\n' "$out" | sed -n 's/.*Average GFLOPS: \([0-9.]*\).*/\1/p' | head -n1)

  if [[ $rc -ne 0 ]] || [[ -z "$avg_ms" ]] || [[ -z "$gflops" ]]; then
    echo "${name},${thr},${KERNEL},," >> "$RESULT_CSV"
    echo "  WARN: exit code $rc (empty metrics = parse failed)" >&2
  else
    echo "${name},${thr},${KERNEL},${avg_ms},${gflops}" >> "$RESULT_CSV"
  fi
done < "$DATASETS"

echo ""
echo "========================================"
echo "Done. Wrote $RESULT_CSV"
