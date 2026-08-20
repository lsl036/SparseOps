#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPARSEOPS_ROOT="${SPARSEOPS_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
SPGEMM_BIN="${SPARSEOPS_ROOT}/build/test_spgemm_hc_lsh"
MATRIX="${SPARSEOPS_ROOT}/data/gupta3/gupta3.mtx"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/l2_fraction.txt}"
PERF_REPEATS=3

export SPARSEOPS_ROOT
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export LC_ALL=C

if ! command -v perf >/dev/null 2>&1; then
    echo "error: perf is not installed or is not in PATH" >&2
    exit 1
fi
if [[ ! -x "${SPGEMM_BIN}" ]]; then
    echo "error: executable not found: ${SPGEMM_BIN}" >&2
    exit 1
fi
if [[ ! -f "${MATRIX}" ]]; then
    echo "error: matrix not found: ${MATRIX}" >&2
    exit 1
fi

CPU_VENDOR="$(awk -F: '/^vendor_id/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' /proc/cpuinfo)"

PERF_OUTPUT="$(mktemp "${TMPDIR:-/tmp}/sparseops-fig7-perf.XXXXXX")"
WORKLOAD_OUTPUT="$(mktemp "${TMPDIR:-/tmp}/sparseops-fig7-workload.XXXXXX")"
trap 'rm -f "${PERF_OUTPUT}" "${WORKLOAD_OUTPUT}"' EXIT

# Return 0 if both symbolic event names are usable with the local perf.
events_available() {
    local all_event="$1"
    local miss_event="$2"
    local events="${all_event}:u,${miss_event}:u"

    : >"${PERF_OUTPUT}"
    if ! perf stat --no-big-num -x ';' -e "${events}" \
        -o "${PERF_OUTPUT}" -- true >/dev/null 2>&1; then
        return 1
    fi
    if grep -Eq '<not (supported|counted)>' "${PERF_OUTPUT}"; then
        return 1
    fi
    return 0
}

# Select a vendor-specific L2 access/miss event pair.
# Intel: prefer the classic Skylake-era names; fall back to Ice Lake demand-data
# events when l2_rqsts.references / l2_rqsts.miss are unavailable (e.g. Xeon 8360).
case "${CPU_VENDOR}" in
    AuthenticAMD)
        L2_ALL_EVENT="l2_cache_accesses_from_dc_misses"
        L2_MISS_EVENT="l2_cache_misses_from_dc_misses"
        ;;
    GenuineIntel)
        L2_ALL_EVENT=""
        L2_MISS_EVENT=""
        if events_available "l2_rqsts.references" "l2_rqsts.miss"; then
            L2_ALL_EVENT="l2_rqsts.references"
            L2_MISS_EVENT="l2_rqsts.miss"
        elif events_available "l2_rqsts.all_demand_data_rd" "l2_rqsts.all_demand_miss"; then
            L2_ALL_EVENT="l2_rqsts.all_demand_data_rd"
            L2_MISS_EVENT="l2_rqsts.all_demand_miss"
            echo "note: using Ice Lake L2 demand-data events" \
                "(l2_rqsts.references / l2_rqsts.miss are unavailable on this CPU)" >&2
        else
            echo "error: no usable Intel L2 perf events found" >&2
            echo "Tried:" >&2
            echo "  l2_rqsts.references, l2_rqsts.miss" >&2
            echo "  l2_rqsts.all_demand_data_rd, l2_rqsts.all_demand_miss" >&2
            cat "${PERF_OUTPUT}" >&2
            exit 1
        fi
        ;;
    *)
        echo "error: unsupported CPU vendor: ${CPU_VENDOR:-unknown}" >&2
        exit 1
        ;;
esac

PERF_EVENTS="${L2_ALL_EVENT}:u,${L2_MISS_EVENT}:u"

# Final validation (also covers the AMD path).
if ! events_available "${L2_ALL_EVENT}" "${L2_MISS_EVENT}"; then
    echo "error: perf cannot use events ${PERF_EVENTS}" >&2
    cat "${PERF_OUTPUT}" >&2
    exit 1
fi

extract_counter() {
    local event_name="$1"
    local perf_file="$2"

    awk -F';' -v expected="${event_name}" '
        {
            name = $3
            sub(/:u$/, "", name)
            if (name == expected) {
                value = $1
                gsub(/[[:space:]]/, "", value)
                print value
                exit
            }
        }
    ' "${perf_file}"
}

mkdir -p "$(dirname -- "${OUTPUT_FILE}")"
: >"${OUTPUT_FILE}"

echo "CPU vendor: ${CPU_VENDOR}"
echo "L2 access event: ${L2_ALL_EVENT}"
echo "L2 miss event: ${L2_MISS_EVENT}"
echo "perf repetitions per fraction: ${PERF_REPEATS}"
echo "Output: ${OUTPUT_FILE}"

for ((fraction_step = 1; fraction_step <= 11; ++fraction_step)); do
    printf -v fraction '%d.%d' "$((fraction_step / 10))" "$((fraction_step % 10))"
    printf '[%02d/11] l2_fraction=%s ... ' "${fraction_step}" "${fraction}"

    : >"${PERF_OUTPUT}"
    : >"${WORKLOAD_OUTPUT}"
    if ! perf stat -r "${PERF_REPEATS}" --no-big-num -x ';' \
        -e "${PERF_EVENTS}" -o "${PERF_OUTPUT}" -- \
        "${SPGEMM_BIN}" \
        "${MATRIX}" \
        "${MATRIX}" \
        --k=64 \
        --bands=16 \
        --hc_v=0 \
        --kernel=3 \
        --l2_fraction="${fraction}" \
        >"${WORKLOAD_OUTPUT}" 2>&1; then
        echo "failed" >&2
        cat "${WORKLOAD_OUTPUT}" >&2
        cat "${PERF_OUTPUT}" >&2
        exit 1
    fi

    l2_all_count="$(extract_counter "${L2_ALL_EVENT}" "${PERF_OUTPUT}")"
    l2_miss_count="$(extract_counter "${L2_MISS_EVENT}" "${PERF_OUTPUT}")"
    if [[ ! "${l2_all_count}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        [[ ! "${l2_miss_count}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "failed to parse perf counters" >&2
        cat "${PERF_OUTPUT}" >&2
        exit 1
    fi

    l2_hit_rate="$(awk -v misses="${l2_miss_count}" -v accesses="${l2_all_count}" '
        BEGIN {
            if (accesses <= 0) {
                exit 1
            }
            printf "%.8f", 1.0 - misses / accesses
        }
    ')" || {
        echo "error: invalid L2 access count: ${l2_all_count}" >&2
        exit 1
    }

    printf '%s %s %s %s\n' \
        "${fraction}" "${l2_miss_count}" "${l2_all_count}" "${l2_hit_rate}" \
        >>"${OUTPUT_FILE}"
    printf 'miss=%s accesses=%s hit_rate=%s\n' \
        "${l2_miss_count}" "${l2_all_count}" "${l2_hit_rate}"
done

echo "Wrote ${OUTPUT_FILE}"
