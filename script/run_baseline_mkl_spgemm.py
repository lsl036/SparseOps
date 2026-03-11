#!/usr/bin/env python3
"""
Run baseline_mkl_spgemm for each dataset in given list file(s).
Command: baseline_mkl_spgemm BASE/name/name.mtx BASE/name/name.mtx  (A=B, square SpGEMM)
Output CSV: mtx_name, average_time, GFLOPS

Usage:
  # From project root, run both testdatasets.txt and alldatasets.txt
  python3 script/run_baseline_mkl_spgemm.py testdatasets.txt alldatasets.txt -o mkl_spgemm_results.csv

  # Single list, custom base dir and binary
  python3 script/run_baseline_mkl_spgemm.py testdatasets.txt --base-dir /data/suitesparse_collection --binary build/baseline_mkl_spgemm -o out.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_mkl_spgemm_output(stdout: str):
    """Extract average_time (ms) and GFLOPS from baseline_mkl_spgemm stdout.
    Returns (average_time, gflops) or (None, None) on failure.
    """
    # e.g. "MKL SpGEMM: 10 iterations, average 37.6841 ms, 14.3024 GFLOPS"
    m = re.search(r"average\s+([\d.]+)\s*ms\s*,\s*([\d.]+)\s*GFLOPS", stdout)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline_mkl_spgemm for datasets in list file(s), record mtx_name, average_time, GFLOPS."
    )
    parser.add_argument(
        "list_files",
        nargs="+",
        help="Dataset list file(s), e.g. testdatasets.txt alldatasets.txt (one name per line)",
    )
    parser.add_argument(
        "--base-dir",
        default="/data/suitesparse_collection",
        help="Base dir for matrices: BASE_DIR/name/name.mtx",
    )
    parser.add_argument(
        "--binary",
        default=None,
        help="Path to baseline_mkl_spgemm (default: build/baseline_mkl_spgemm relative to script)",
    )
    parser.add_argument(
        "-o", "--output",
        default="mkl_spgemm_results.csv",
        help="Output CSV path (default: mkl_spgemm_results.csv)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Pass --iterations N to baseline_mkl_spgemm (default: 10)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    base_dir = Path(args.base_dir)
    binary = args.binary
    if not binary:
        binary = str(root / "build" / "baseline_mkl_spgemm")
    binary = str(Path(binary).resolve())
    if not Path(binary).is_file():
        print(f"Error: binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    # Collect unique dataset names from all list files (preserve order, first occurrence)
    seen = set()
    names = []
    for list_path in args.list_files:
        p = Path(list_path)
        if not p.is_file():
            print(f"Warning: list file not found: {p}", file=sys.stderr)
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").strip().splitlines():
            name = line.split("#")[0].strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)

    if not names:
        print("Error: no dataset names found from list files.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    fieldnames = ["mtx_name", "average_time", "GFLOPS"]
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, name in enumerate(names, 1):
            mtx = base_dir / name / f"{name}.mtx"
            if not mtx.is_file():
                print(f"[{i}/{len(names)}] Skip {name}: not found {mtx}", file=sys.stderr)
                writer.writerow({"mtx_name": name, "average_time": "", "GFLOPS": ""})
                csvfile.flush()
                continue

            cmd = [binary, str(mtx), str(mtx), f"--iterations={args.iterations}"]
            print(f"[{i}/{len(names)}] Run {name} ...", flush=True)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                stdout = result.stdout or ""
                if result.returncode != 0:
                    print(f"  FAIL: exit {result.returncode}", file=sys.stderr)
                    writer.writerow({"mtx_name": name, "average_time": "", "GFLOPS": ""})
                else:
                    avg_ms, gflops = parse_mkl_spgemm_output(stdout)
                    if avg_ms is not None and gflops is not None:
                        writer.writerow({"mtx_name": name, "average_time": avg_ms, "GFLOPS": gflops})
                        print(f"  {avg_ms:.4f} ms, {gflops:.4f} GFLOPS", flush=True)
                    else:
                        writer.writerow({"mtx_name": name, "average_time": "", "GFLOPS": ""})
                        print(f"  parse failed", file=sys.stderr)
            except subprocess.TimeoutExpired:
                print(f"  timeout", file=sys.stderr)
                writer.writerow({"mtx_name": name, "average_time": "", "GFLOPS": ""})
            except Exception as e:
                print(f"  error: {e}", file=sys.stderr)
                writer.writerow({"mtx_name": name, "average_time": "", "GFLOPS": ""})
            csvfile.flush()

    print(f"\nResults written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
