#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPARSEOPS_ROOT="${SPARSEOPS_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
ACCUMULATOR_BIN="${SPARSEOPS_ROOT}/build/Microbench_accumulator"
OUTPUT_FILE="${SCRIPT_DIR}/accumulator.txt"

export SPARSEOPS_ROOT
export OMP_NUM_THREADS=1

if [[ ! -x "${ACCUMULATOR_BIN}" ]]; then
    echo "error: executable not found: ${ACCUMULATOR_BIN}" >&2
    exit 1
fi

"${ACCUMULATOR_BIN}" >"${OUTPUT_FILE}" 2>&1
echo "Wrote ${OUTPUT_FILE}"
