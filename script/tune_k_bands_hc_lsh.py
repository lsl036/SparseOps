#!/usr/bin/env python3
"""
Auto-tune k and bands for test_spgemm_hc_lsh.
- Runs over a list of datasets (matrix names) under a suite sparse base dir.
- k in [32, 256] step 32; bands such that r = k/bands in [3, 6] and k % bands == 0.
- Records: mtxname, k, bands, r, genPairs_time, HC_time, Format_Conversion_time, LeSpGEMM_VLength_time.

Usage:
  # One or more datasets; matrix path = BASE_DIR/NAME/NAME.mtx
  python3 script/tune_k_bands_hc_lsh.py pdb1HYS scircuit --base-dir /data/suitesparse_collection -o results.csv
  # Run from build dir or set --binary to path of test_spgemm_hc_lsh

  # Datasets from file (one name per line)
  python3 script/tune_k_bands_hc_lsh.py --datasets-file datasets.txt --binary ./build/test_spgemm_hc_lsh -o out.csv

  # Optional: fewer iterations per run
  python3 script/tune_k_bands_hc_lsh.py pdb1HYS --extra "--iterations=5"
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


def run_one(binary, mtx_path, k, bands, extra_args):
    """Run test_spgemm_hc_lsh once. Returns (parsed_times_dict, stderr_or_empty)."""
    cmd = [
        binary,
        mtx_path,
        mtx_path,
        f"--k={k}",
        f"--bands={bands}",
        *extra_args,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if result.returncode != 0:
            return None, stderr.strip() or f"exit code {result.returncode}"
        parsed = parse_run_output(stdout)
        return parsed, stderr.strip()
    except subprocess.TimeoutExpired:
        return None, "timeout"
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
        "--extra",
        default="",
        help="Extra args passed to binary (e.g. --iterations=5 --cluster_size=8)",
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

    extra_args = args.extra.split() if args.extra else []
    k_bands_list = list(k_bands_candidates())

    fieldnames = [
        "mtxname",
        "k",
        "bands",
        "r",
        "genPairs_time",
        "HC_time",
        "Format_Conversion_time",
        "LeSpGEMM_VLength_time",
    ]

    rows = []
    for name in datasets:
        mtx_path = get_matrix_path(base_dir, name)
        if not os.path.isfile(mtx_path):
            print(f"Skip (file not found): {mtx_path}", file=sys.stderr)
            continue
        for k, bands in k_bands_list:
            r = k // bands
            print(f"Run {name} k={k} bands={bands} r={r} ...", flush=True)
            parsed, err = run_one(binary, mtx_path, k, bands, extra_args)
            if parsed is None:
                print(f"  FAIL: {err}", file=sys.stderr)
                rows.append({
                    "mtxname": name,
                    "k": k,
                    "bands": bands,
                    "r": r,
                    "genPairs_time": "",
                    "HC_time": "",
                    "Format_Conversion_time": "",
                    "LeSpGEMM_VLength_time": "",
                    "_error": err,
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
                })
                print(f"  genPairs={parsed['genPairs_time']:.2f} HC={parsed['HC_time']:.2f} "
                      f"Convert={parsed['Format_Conversion_time']:.2f} SpGEMM={parsed['LeSpGEMM_VLength_time']:.2f} ms")

    out_path = args.output
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
