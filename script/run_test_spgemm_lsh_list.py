#!/usr/bin/env python3
"""
Batch-run SpGEMM using pre-recorded LSH reordering (.perm/.offsets) by calling test_spgemm_lsh.

Input list file format (whitespace-separated):
  mtx_name   k   bands   [optional rest ignored]

Note: k/bands are not required by test_spgemm_lsh at runtime; they are kept for bookkeeping.

Usage:
  # from repo root
  python3 script/run_test_spgemm_lsh_list.py runable_casesets.txt

  # from build dir
  python3 ../script/run_test_spgemm_lsh_list.py ../runable_casesets.txt --binary ./test_spgemm_lsh
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_run_output(text: str):
    out = {}
    m = re.search(r"Average time \\(LeSpGEMM_VLength kernel=3\\):\\s*([\\d.]+)\\s*ms", text)
    if not m:
        return None
    out["avg_ms"] = float(m.group(1))
    m = re.search(r"Average GFLOPS:\\s*([\\d.]+)", text)
    out["gflops"] = float(m.group(1)) if m else None
    m = re.search(r"\\[mixed_acc\\]\\s*dense clusters:\\s*([0-9]+)\\s*/\\s*([0-9]+)", text)
    out["dense_clusters"] = int(m.group(1)) if m else None
    out["total_clusters"] = int(m.group(2)) if m else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run test_spgemm_lsh for each dataset name in a list file."
    )
    parser.add_argument(
        "list_file",
        help="File with lines: dataset_name  k  bands  (tab or space separated)",
    )
    parser.add_argument(
        "--base-dir",
        default="/data/suitesparse_collection",
        help="Base dir for matrices: BASE_DIR/name/name.mtx",
    )
    parser.add_argument(
        "--binary",
        default="./test_spgemm_lsh",
        help="Path to test_spgemm_lsh executable",
    )
    parser.add_argument(
        "--lsh-order-dir",
        default="/data2/linshengle_data/SpGEMM-Reordering/lsh_order",
        help="Directory containing <name>.perm and <name>.offsets",
    )
    parser.add_argument(
        "--iterations",
        default="10",
        help="Iterations for timing (default 10)",
    )
    parser.add_argument(
        "--l2-fraction",
        default="-1",
        help="l2_fraction passed to kernel=3 (default -1 = MIXED_L2_FRACTION)",
    )
    parser.add_argument(
        "--precision",
        default="64",
        help="32|64 (default 64)",
    )
    parser.add_argument(
        "--threads",
        default="",
        help="OMP threads to pass via --threads (optional)",
    )
    parser.add_argument(
        "--sort",
        default="0",
        help="0|1 sort output columns (default 0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Append stdout/stderr of each run to FILE (optional)",
    )
    parser.add_argument(
        "-c",
        "--csv",
        metavar="FILE",
        help="Write results to CSV (default: <list_file_stem>_spgemm_lsh_results.csv)",
    )
    args = parser.parse_args()

    list_path = Path(args.list_file)
    if not list_path.is_file():
        print(f"Error: list file not found: {list_path}", file=sys.stderr)
        return 1

    base_dir = args.base_dir.rstrip("/")
    lsh_order_dir = args.lsh_order_dir
    binary = args.binary
    if not os.path.isabs(binary):
        binary = str(Path(binary).resolve())
    if not os.path.isfile(binary):
        alt = Path(__file__).resolve().parent.parent / "build" / "test_spgemm_lsh"
        if alt.is_file():
            binary = str(alt)
    if not os.path.isfile(binary):
        print(f"Error: binary not found: {args.binary}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["OMP_PLACES"] = "cores"
    env["OMP_PROC_BIND"] = "spread"

    lines = list_path.read_text(encoding="utf-8", errors="replace").splitlines()
    log_file = open(args.output, "a", encoding="utf-8") if args.output else None

    csv_path = Path(args.csv) if args.csv else list_path.parent / (list_path.stem + "_spgemm_lsh_results.csv")
    fieldnames = [
        "mtxname", "k", "bands",
        "avg_ms", "gflops",
        "dense_clusters", "total_clusters",
        "error",
    ]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    total = 0
    failed = 0
    try:
        for i, line in enumerate(lines, 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) < 1:
                continue
            name = parts[0]
            k = parts[1] if len(parts) >= 2 else ""
            bands = parts[2] if len(parts) >= 3 else ""

            mtx = os.path.join(base_dir, name, f"{name}.mtx")
            perm_path = os.path.join(lsh_order_dir, f"{name}.perm")
            offsets_path = os.path.join(lsh_order_dir, f"{name}.offsets")

            row = {
                "mtxname": name, "k": k, "bands": bands,
                "avg_ms": "", "gflops": "",
                "dense_clusters": "", "total_clusters": "",
                "error": "",
            }

            if not os.path.isfile(mtx):
                row["error"] = f"matrix not found: {mtx}"
                writer.writerow(row)
                csv_file.flush()
                failed += 1
                total += 1
                print(f"Skip {name}: matrix not found {mtx}", file=sys.stderr)
                continue
            if not os.path.isfile(perm_path) or not os.path.isfile(offsets_path):
                row["error"] = "missing perm/offsets (run record_permutation_lsh first)"
                writer.writerow(row)
                csv_file.flush()
                failed += 1
                total += 1
                print(f"Skip {name}: missing {perm_path} or {offsets_path}", file=sys.stderr)
                continue

            cmd = [
                binary,
                mtx,
                mtx,
                f"--lsh-order-dir={lsh_order_dir}",
                f"--iterations={args.iterations}",
                f"--l2_fraction={args.l2_fraction}",
                f"--precision={args.precision}",
                f"--sort={args.sort}",
            ]
            if args.threads:
                cmd.append(f"--threads={args.threads}")

            print(f"[{i}/{len(lines)}] SpGEMM {name} ...", flush=True)
            total += 1
            try:
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=None,
                )
                stdout = result.stdout or ""
                stderr = result.stderr or ""

                if log_file:
                    log_file.write(f"\n===== {name} =====\n")
                    log_file.write("CMD: " + " ".join(cmd) + "\n")
                    log_file.write(stdout)
                    if stderr:
                        log_file.write(stderr)
                    log_file.write(f"\nExit code: {result.returncode}\n")
                    log_file.flush()

                if result.returncode != 0:
                    failed += 1
                    row["error"] = (stderr.strip() or f"exit code {result.returncode}")[:500]
                    print(f"  FAIL: exit code {result.returncode}", file=sys.stderr)
                else:
                    parsed = parse_run_output(stdout)
                    if not parsed:
                        failed += 1
                        row["error"] = "parse_failed"
                        print("  FAIL: parse_failed", file=sys.stderr)
                    else:
                        row["avg_ms"] = parsed["avg_ms"]
                        row["gflops"] = parsed["gflops"] if parsed["gflops"] is not None else ""
                        row["dense_clusters"] = parsed["dense_clusters"] if parsed["dense_clusters"] is not None else ""
                        row["total_clusters"] = parsed["total_clusters"] if parsed["total_clusters"] is not None else ""
                        print("  OK", flush=True)

            except Exception as e:
                failed += 1
                row["error"] = str(e)[:500]
                print(f"  FAIL: {e}", file=sys.stderr)

            writer.writerow(row)
            csv_file.flush()

        print(f"\nDone: {total - failed}/{total} passed, {failed} failed. CSV: {csv_path}", file=sys.stderr)
    finally:
        if log_file:
            log_file.close()
        csv_file.close()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

