#!/usr/bin/env python3
"""
Auto-tune k and bands for test_spgemm_hc_lsh.
- Runs over a list of datasets (matrix names) under a suite sparse base dir.
- k in [32, 256] step 32; bands such that r = k/bands in [3, 6] and k % bands == 0.
- Records: mtxname, k, bands, r, genPairs_time, HC_time, Format_Conversion_time, LeSpGEMM_VLength_time.

Usage:
  # Script sets OMP_PLACES=cores and OMP_PROC_BIND=spread for each run.
  # One or more datasets; matrix path = BASE_DIR/NAME/NAME.mtx
  python3 script/tune_k_bands_hc_lsh.py pdb1HYS scircuit --base-dir /data/suitesparse_collection -o results.csv
  # Run from build dir or set --binary to path of test_spgemm_hc_lsh

  # Datasets from file (one name per line)
  python3 script/tune_k_bands_hc_lsh.py --datasets-file datasets.txt --binary ./build/test_spgemm_hc_lsh -o out.csv

  # Optional: fewer iterations per run
  python3 script/tune_k_bands_hc_lsh.py pdb1HYS --extra "--iterations=5"

  # Use mixed-accumulator kernel (kernel=3)
  python3 script/tune_k_bands_hc_lsh.py pdb1HYS --extra "--kernel=3"
"""

from __future__ import print_function

import argparse
import csv
import os
import re
import subprocess
import sys


def get_matrix_path(base_dir: str, name: str) -> str:
    """Path to matrix: base_dir/name/name.mtx"""
    return os.path.join(base_dir, name, f"{name}.mtx")


def k_bands_candidates():
    """Yield (k, bands) with k in [32,256] step 32, r=k/bands in [3,6], k%bands==0."""
    for k in range(32, 256 + 1, 32):
        for b in range(1, k + 1):
            if k % b != 0:
                continue
            r = k // b
            if 3 <= r <= 6:
                yield (k, b)


def parse_run_output(text: str):
    """Extract times (ms) from program stdout. Returns dict or None on failure."""
    out = {}
    # genPairs time: 123.45 ms
    m = re.search(r"genPairs time:\s*([\d.]+)\s*ms", text)
    if not m:
        return None
    out["genPairs_time"] = float(m.group(1))
    # HC time: 12.34 ms
    m = re.search(r"HC time:\s*([\d.]+)\s*ms", text)
    if not m:
        return None
    out["HC_time"] = float(m.group(1))
    # Format Conversion time: 5.67 ms
    m = re.search(r"Format Conversion time:\s*([\d.]+)\s*ms", text)
    if not m:
        return None
    out["Format_Conversion_time"] = float(m.group(1))
    # Average time (LeSpGEMM_VLength): 89.01 ms
    m = re.search(r"Average time \(LeSpGEMM_VLength\):\s*([\d.]+)\s*ms", text)
    if not m:
        return None
    out["LeSpGEMM_VLength_time"] = float(m.group(1))
    return out


def run_one(binary, mtx_path, k, bands, extra_args, timeout_sec):
    """Run test_spgemm_hc_lsh once. Returns (parsed_times_dict, error_msg). timeout_sec=None means no limit."""
    cmd = [
        binary,
        mtx_path,
        mtx_path,
        "--k=%d" % k,
        "--bands=%d" % bands,
        *extra_args,
    ]
    env = os.environ.copy()
    env["OMP_PLACES"] = "cores"
    env["OMP_PROC_BIND"] = "spread"
    kwargs = {"capture_output": True, "text": True, "env": env}
    if timeout_sec is not None:
        kwargs["timeout"] = timeout_sec
    try:
        result = subprocess.run(cmd, **kwargs)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if result.returncode != 0:
            return None, stderr.strip() or ("exit code %d" % result.returncode)
        parsed = parse_run_output(stdout)
        if parsed is None:
            return None, "parse_failed"
        return parsed, ""
    except subprocess.TimeoutExpired:
        return None, "timeout(%ds)" % (timeout_sec or 0)
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Tune k and bands for test_spgemm_hc_lsh over multiple datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names (e.g. pdb1HYS). Matrix path: BASE_DIR/NAME/NAME.mtx",
    )
    parser.add_argument(
        "--datasets-file",
        metavar="FILE",
        help="Read dataset names from file (one name per line, # ignored)",
    )
    parser.add_argument(
        "--base-dir",
        default="/data/suitesparse_collection",
        help="Base directory for SuiteSparse matrices; matrix path = BASE_DIR/NAME/NAME.mtx",
    )
    parser.add_argument(
        "--binary",
        default="./test_spgemm_hc_lsh",
        help="Path to test_spgemm_hc_lsh executable (default: ./test_spgemm_hc_lsh)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tune_k_bands_results.csv",
        help="Output CSV path (default: tune_k_bands_results.csv)",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=None,
        metavar="N",
        help="Pass --kernel=N to binary (e.g. 3 for mixed-accumulator). Avoids quoting with --extra.",
    )
    parser.add_argument(
        "--extra",
        nargs="?",
        default="",
        metavar="ARGS",
        help="Extra args passed to binary (e.g. '--iterations=5 --cluster_size=8'). For kernel use --kernel 3.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        metavar="SEC",
        help="Max run time per (dataset, k, bands) in seconds; 0 = no limit (default: 600)",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.rstrip("/")
    binary = os.path.abspath(args.binary)
    if not os.path.isfile(binary):
        # try relative to script/cwd
        if not os.path.isabs(args.binary):
            alt = os.path.join(os.path.dirname(__file__) or ".", "..", "build", "test_spgemm_hc_lsh")
            if os.path.isfile(alt):
                binary = os.path.abspath(alt)
        if not os.path.isfile(binary):
            print(f"Error: binary not found: {args.binary}", file=sys.stderr)
            sys.exit(1)

    # Resolve dataset list
    if args.datasets_file:
        with open(args.datasets_file) as f:
            names_from_file = [
                line.split("#")[0].strip()
                for line in f
                if line.split("#")[0].strip()
            ]
        datasets = names_from_file
    else:
        datasets = args.datasets
    if not datasets:
        print("Error: provide datasets as arguments or --datasets-file FILE", file=sys.stderr)
        sys.exit(1)

    extra_args = []
    if args.kernel is not None:
        extra_args.append("--kernel=%d" % args.kernel)
    if args.extra:
        extra_args.extend(args.extra.split())
    k_bands_list = list(k_bands_candidates())

    timeout_sec = args.timeout if args.timeout > 0 else None  # None = no limit
    if timeout_sec is not None:
        print("Per-run timeout: %d s (use --timeout 0 for no limit)" % timeout_sec, file=sys.stderr)
    else:
        print("Per-run timeout: none", file=sys.stderr)

    fieldnames = [
        "mtxname",
        "k",
        "bands",
        "r",
        "genPairs_time",
        "HC_time",
        "Format_Conversion_time",
        "LeSpGEMM_VLength_time",
        "error",
    ]

    rows = []
    for name in datasets:
        mtx_path = get_matrix_path(base_dir, name)
        if not os.path.isfile(mtx_path):
            print("Skip (file not found): %s" % mtx_path, file=sys.stderr)
            rows.append({
                "mtxname": name,
                "k": "",
                "bands": "",
                "r": "",
                "genPairs_time": "",
                "HC_time": "",
                "Format_Conversion_time": "",
                "LeSpGEMM_VLength_time": "",
                "error": "file_not_found",
            })
            continue
        for k, bands in k_bands_list:
            r = k // bands
            print("Run %s k=%d bands=%d r=%d ..." % (name, k, bands, r), flush=True)
            parsed, err = run_one(binary, mtx_path, k, bands, extra_args, timeout_sec)
            if parsed is None:
                print("  FAIL: %s" % err, file=sys.stderr)
                rows.append({
                    "mtxname": name,
                    "k": k,
                    "bands": bands,
                    "r": r,
                    "genPairs_time": "",
                    "HC_time": "",
                    "Format_Conversion_time": "",
                    "LeSpGEMM_VLength_time": "",
                    "error": err,
                })
            else:
                rows.append({
                    "mtxname": name,
                    "k": k,
                    "bands": bands,
                    "r": r,
                    "genPairs_time": parsed["genPairs_time"],
                    "HC_time": parsed["HC_time"],
                    "Format_Conversion_time": parsed["Format_Conversion_time"],
                    "LeSpGEMM_VLength_time": parsed["LeSpGEMM_VLength_time"],
                    "error": "",
                })
                print("  genPairs=%.2f HC=%.2f Convert=%.2f SpGEMM=%.2f ms" % (
                    parsed["genPairs_time"], parsed["HC_time"],
                    parsed["Format_Conversion_time"], parsed["LeSpGEMM_VLength_time"]))

    out_path = args.output
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for r in rows if r.get("error") == "" and r.get("genPairs_time") != "")
    print("Wrote %d rows to %s (%d with results, %d failed/skipped)" % (
        len(rows), out_path, n_ok, len(rows) - n_ok))


if __name__ == "__main__":
    main()
