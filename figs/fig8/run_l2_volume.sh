#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPARSEOPS_ROOT="${SPARSEOPS_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
LOCAL_LIKWID="${SCRIPT_DIR}/likwid-5.5.1/likwid-perfctr"
SPGEMM_BIN="${SPARSEOPS_ROOT}/build/test_spgemm_lsh"
MATRIX="${SPARSEOPS_ROOT}/data/gupta3/gupta3.mtx"
LSH_ORDER_DIR="${SPARSEOPS_ROOT}/data/reordering/lsh_order"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/l2_volume.txt}"
SPGEMM_ITERATIONS="${SPGEMM_ITERATIONS:-10}"

export SPARSEOPS_ROOT
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export LC_ALL=C

if [[ -n "${LIKWID_PERFCTR:-}" ]]; then
    LIKWID_BIN="${LIKWID_PERFCTR}"
elif [[ -x "${LOCAL_LIKWID}" ]]; then
    LIKWID_BIN="${LOCAL_LIKWID}"
elif command -v likwid-perfctr >/dev/null 2>&1; then
    LIKWID_BIN="$(command -v likwid-perfctr)"
else
    echo "error: no executable likwid-perfctr was found" >&2
    if [[ -f "${LOCAL_LIKWID}" ]]; then
        echo "Local LIKWID exists but is not executable: ${LOCAL_LIKWID}" >&2
        echo "Rebuild it with 'make distclean && make -j ACCESSMODE=perf_event && make ACCESSMODE=perf_event local'." >&2
    fi
    exit 1
fi

if [[ ! -x "${LIKWID_BIN}" ]]; then
    echo "error: likwid-perfctr is not executable: ${LIKWID_BIN}" >&2
    exit 1
fi

USING_LOCAL_LIKWID=0
LOCAL_LIKWID_ROOT=""
if [[ "${LIKWID_BIN}" == "${LOCAL_LIKWID}" ]]; then
    USING_LOCAL_LIKWID=1
    LOCAL_LIKWID_ROOT="$(dirname -- "${LOCAL_LIKWID}")"
    if [[ ! -d "${LOCAL_LIKWID_ROOT}/groups" ]] ||
        [[ ! -e "${LOCAL_LIKWID_ROOT}/liblikwid-lua.so.5.5" ]]; then
        echo "error: local LIKWID build is incomplete: ${LOCAL_LIKWID_ROOT}" >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="${LOCAL_LIKWID_ROOT}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ ! -x "${SPGEMM_BIN}" ]]; then
    echo "error: executable not found: ${SPGEMM_BIN}" >&2
    exit 1
fi
if [[ ! -f "${MATRIX}" ]]; then
    echo "error: matrix not found: ${MATRIX}" >&2
    exit 1
fi
if [[ ! -f "${LSH_ORDER_DIR}/gupta3.perm" ]] ||
    [[ ! -f "${LSH_ORDER_DIR}/gupta3.offsets" ]]; then
    echo "error: gupta3 LSH reordering files are missing from ${LSH_ORDER_DIR}" >&2
    exit 1
fi
if ! [[ "${SPGEMM_ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: SPGEMM_ITERATIONS must be a positive integer" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to parse LIKWID CSV output" >&2
    exit 1
fi

physical_cpu_list() {
    lscpu -p=CPU,CORE,SOCKET,ONLINE | awk -F, '
        !/^#/ && $4 == "Y" {
            key = $2 ":" $3
            if (!seen[key]++) {
                cpus[count++] = $1
            }
        }
        END {
            for (i = 0; i < count; ++i) {
                printf "%s%s", i == 0 ? "" : ",", cpus[i]
            }
            print ""
        }
    '
}

count_cpu_list() {
    awk -F, '
        {
            count = 0
            for (i = 1; i <= NF; ++i) {
                parts_count = split($i, parts, "-")
                if (parts_count == 2) {
                    count += parts[2] - parts[1] + 1
                } else {
                    count += 1
                }
            }
            print count
        }
    ' <<<"$1"
}

CPU_VENDOR="$(awk -F: '/^vendor_id/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' /proc/cpuinfo)"
LIKWID_EXTRA_ARGS=()
METRIC_CANDIDATES=()
case "${CPU_VENDOR}" in
    GenuineIntel)
        DEFAULT_CPU_LIST="0-27"
        VOLUME_KIND="L2 eviction traffic"
        PARSER_MODE="metric"
        LIKWID_GROUP="L3"
        METRIC_CANDIDATES=(
            "L3|MEM evict data volume [GBytes]"
            "L3 evict data volume [GBytes]"
        )
        ;;
    AuthenticAMD)
        DEFAULT_CPU_LIST="$(physical_cpu_list)"
        VOLUME_KIND="L2 fill/miss traffic"
        PARSER_MODE="amd_raw"
        LIKWID_GROUP="L2_PF_HIT_IN_L3:PMC0,L2_PF_MISS_IN_L3:PMC1,L2_CACHE_MISS_AFTER_L1_MISS:PMC2"
        LIKWID_EXTRA_ARGS=(--execpid)
        ;;
    *)
        echo "error: unsupported CPU vendor: ${CPU_VENDOR:-unknown}" >&2
        exit 1
        ;;
esac

CPU_LIST="${CPU_LIST:-${DEFAULT_CPU_LIST}}"
THREADS="${THREADS:-$(count_cpu_list "${CPU_LIST}")}"
if [[ -z "${CPU_LIST}" ]] || ! [[ "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: invalid CPU_LIST or THREADS setting" >&2
    exit 1
fi

parse_metric_stats() {
    local csv_file="$1"
    shift

    python3 - "${csv_file}" "$@" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
candidates = sys.argv[2:]

with open(csv_path, newline="", encoding="utf-8", errors="replace") as stream:
    rows = list(csv.reader(stream))

available_metrics = []
for index, row in enumerate(rows):
    if len(row) < 2 or row[0] != "TABLE" or "Metric STAT" not in row[1]:
        continue
    if index + 1 >= len(rows) or not rows[index + 1]:
        continue

    header = rows[index + 1]
    available_metrics.extend(
        value.removesuffix(" STAT") for value in header[1:] if value
    )
    for candidate in candidates:
        target = candidate + " STAT"
        if target not in header:
            continue

        column = header.index(target)
        stats = {}
        for values in rows[index + 2 :]:
            if values and values[0] == "TABLE":
                break
            if values and values[0] in {"Sum", "Min", "Max", "Avg"}:
                if column < len(values):
                    stats[values[0]] = values[column].strip()
        if all(name in stats for name in ("Sum", "Min", "Max", "Avg")):
            print(
                "\t".join(
                    (
                        candidate,
                        stats["Sum"],
                        stats["Min"],
                        stats["Max"],
                        stats["Avg"],
                    )
                )
            )
            raise SystemExit(0)

print(
    "error: none of the requested metrics were found: " + ", ".join(candidates),
    file=sys.stderr,
)
if available_metrics:
    print("available metrics: " + ", ".join(available_metrics), file=sys.stderr)
raise SystemExit(1)
PY
}

parse_amd_fill_stats() {
    local csv_file="$1"

    python3 - "${csv_file}" <<'PY'
import csv
import math
import sys

csv_path = sys.argv[1]
events = (
    "L2_PF_HIT_IN_L3",
    "L2_PF_MISS_IN_L3",
    "L2_CACHE_MISS_AFTER_L1_MISS",
)

with open(csv_path, newline="", encoding="utf-8", errors="replace") as stream:
    rows = list(csv.reader(stream))

for index, row in enumerate(rows):
    if len(row) < 3 or row[0] != "TABLE" or not row[1].endswith(" Raw"):
        continue
    if index + 1 >= len(rows):
        continue

    header = rows[index + 1]
    thread_columns = [
        column
        for column, value in enumerate(header)
        if value.startswith("HWThread ")
    ]
    if not thread_columns:
        continue

    counters = {}
    for values in rows[index + 2 :]:
        if values and values[0] == "TABLE":
            break
        if not values or values[0] not in events:
            continue
        try:
            counters[values[0]] = [float(values[column]) for column in thread_columns]
        except (IndexError, ValueError) as error:
            print(f"error: invalid LIKWID counter row: {values}: {error}", file=sys.stderr)
            raise SystemExit(1)

    if not all(event in counters for event in events):
        continue

    volumes = [
        sum(counters[event][thread] for event in events) * 64.0e-9
        for thread in range(len(thread_columns))
    ]
    if not volumes or not all(math.isfinite(value) for value in volumes):
        print("error: invalid AMD L2 fill/miss volumes", file=sys.stderr)
        raise SystemExit(1)

    stats = (sum(volumes), min(volumes), max(volumes), sum(volumes) / len(volumes))
    print(
        "\t".join(
            (
                "L2 fill/miss traffic volume [GBytes]",
                *(f"{value:.12g}" for value in stats),
            )
        )
    )
    raise SystemExit(0)

print("error: AMD L2 fill/miss event rows were not found", file=sys.stderr)
raise SystemExit(1)
PY
}

is_number() {
    [[ "$1" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]
}

mkdir -p "$(dirname -- "${OUTPUT_FILE}")"
RESULT_OUTPUT="$(mktemp "${OUTPUT_FILE}.tmp.XXXXXX")"
printf 'l2_fraction SUM Min Max Avg\n' >"${RESULT_OUTPUT}"
LIKWID_OUTPUT="$(mktemp "${TMPDIR:-/tmp}/sparseops-fig7-likwid.XXXXXX.csv")"
WORKLOAD_OUTPUT="$(mktemp "${TMPDIR:-/tmp}/sparseops-fig7-workload.XXXXXX")"
LIKWID_CONFIG=""
if ((USING_LOCAL_LIKWID)); then
    LIKWID_CONFIG="$(mktemp "${TMPDIR:-/tmp}/sparseops-fig7-likwid.XXXXXX.cfg")"
    printf 'groupPath = %s\ndaemon_mode = perf_event\n' \
        "${LOCAL_LIKWID_ROOT}/groups" >"${LIKWID_CONFIG}"
    export LIKWID_CFG_FILE="${LIKWID_CONFIG}"
fi

cleanup() {
    rm -f "${RESULT_OUTPUT}" "${LIKWID_OUTPUT}" "${WORKLOAD_OUTPUT}"
    if [[ -n "${LIKWID_CONFIG}" ]]; then
        rm -f "${LIKWID_CONFIG}"
    fi
}
trap cleanup EXIT

echo "CPU vendor: ${CPU_VENDOR}"
echo "Volume metric kind: ${VOLUME_KIND}"
echo "LIKWID: ${LIKWID_BIN}"
echo "CPU list: ${CPU_LIST}"
echo "SpGEMM threads: ${THREADS}"
echo "SpGEMM iterations: ${SPGEMM_ITERATIONS}"
echo "Output: ${OUTPUT_FILE}"

SELECTED_METRIC=""
for ((fraction_step = 1; fraction_step <= 11; ++fraction_step)); do
    printf -v fraction '%d.%d' "$((fraction_step / 10))" "$((fraction_step % 10))"
    printf '[%02d/11] l2_fraction=%s ... ' "${fraction_step}" "${fraction}"

    : >"${LIKWID_OUTPUT}"
    : >"${WORKLOAD_OUTPUT}"
    if ! "${LIKWID_BIN}" \
        "${LIKWID_EXTRA_ARGS[@]}" \
        -C "${CPU_LIST}" \
        -g "${LIKWID_GROUP}" \
        --stats \
        -O \
        -o "${LIKWID_OUTPUT}" \
        "${SPGEMM_BIN}" \
        "${MATRIX}" \
        "${MATRIX}" \
        --lsh-order-dir="${LSH_ORDER_DIR}" \
        --threads="${THREADS}" \
        --iterations="${SPGEMM_ITERATIONS}" \
        --l2_fraction="${fraction}" \
        >"${WORKLOAD_OUTPUT}" 2>&1; then
        echo "failed" >&2
        cat "${WORKLOAD_OUTPUT}" >&2
        cat "${LIKWID_OUTPUT}" >&2
        exit 1
    fi

    if [[ "${PARSER_MODE}" == "metric" ]]; then
        parser_command=(parse_metric_stats "${LIKWID_OUTPUT}" "${METRIC_CANDIDATES[@]}")
    else
        parser_command=(parse_amd_fill_stats "${LIKWID_OUTPUT}")
    fi
    if ! parsed_stats="$("${parser_command[@]}")"; then
        echo "failed to parse LIKWID output" >&2
        cat "${WORKLOAD_OUTPUT}" >&2
        cat "${LIKWID_OUTPUT}" >&2
        exit 1
    fi
    IFS=$'\t' read -r metric_name volume_sum volume_min volume_max volume_avg \
        <<<"${parsed_stats}"

    for value in "${volume_sum}" "${volume_min}" "${volume_max}" "${volume_avg}"; do
        if ! is_number "${value}"; then
            echo "error: LIKWID returned a non-numeric statistic: ${value}" >&2
            exit 1
        fi
    done
    if [[ -z "${SELECTED_METRIC}" ]]; then
        SELECTED_METRIC="${metric_name}"
        echo
        echo "Selected LIKWID metric: ${SELECTED_METRIC}"
        printf '[%02d/11] l2_fraction=%s ... ' "${fraction_step}" "${fraction}"
    elif [[ "${metric_name}" != "${SELECTED_METRIC}" ]]; then
        echo "error: LIKWID metric changed between runs" >&2
        exit 1
    fi

    printf '%s %s %s %s %s\n' \
        "${fraction}" "${volume_sum}" "${volume_min}" "${volume_max}" "${volume_avg}" \
        >>"${RESULT_OUTPUT}"
    printf 'sum=%s min=%s max=%s avg=%s\n' \
        "${volume_sum}" "${volume_min}" "${volume_max}" "${volume_avg}"
done

mv -f "${RESULT_OUTPUT}" "${OUTPUT_FILE}"
echo "Wrote ${OUTPUT_FILE}"
