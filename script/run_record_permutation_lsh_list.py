#!/usr/bin/env python3
"""
Batch-generate LSH+HC reordering files (.perm/.offsets) by calling record_permutation_lsh.

Input list file format (whitespace-separated):
  mtx_name   k   bands   [optional rest ignored]

It writes outputs to:
  OUT_DIR/<mtx_name>.perm
  OUT_DIR/<mtx_name>.offsets

Usage:
  # from repo root
  python3 script/run_record_permutation_lsh_list.py runable_casesets.txt

  # from build dir
  python3 ../script/run_record_permutation_lsh_list.py ../runable_casesets.txt --binary ./record_permutation_lsh
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run record_permutation_lsh for each (name,k,bands) in a list file."
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
        default="./record_permutation_lsh",
        help="Path to record_permutation_lsh executable",
    )
    parser.add_argument(
        "--out-dir",
        default="/data/linshengle_data/SpGEMM-Reordering/lsh_order",
        help="Output dir for <name>.perm and <name>.offsets",
    )
    parser.add_argument(
        "--hc-v",
        default="0",
        help="hc_v to pass (0|1|2). Default 0 (same as most tests).",
    )
    parser.add_argument(
        "--cluster-size",
        default="8",
        help="cluster_size to pass (default 8)",
    )
    parser.add_argument(
        "--precision",
        default="64",
        help="32|64 (default 64)",
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
        help="Write results to CSV (default: <list_file_stem>_record_perm_results.csv)",
    )
    parser.add_argument(
        "--record-maxmem",
        action="store_true",
        help="If set, run command with /usr/bin/time -v and record Maximum resident set size.",
    )
    args = parser.parse_args()

    list_path = Path(args.list_file)
    if not list_path.is_file():
        print(f"Error: list file not found: {list_path}", file=sys.stderr)
        return 1

    base_dir = args.base_dir.rstrip("/")
    out_dir = args.out_dir
    binary = args.binary
    if not os.path.isabs(binary):
        binary = str(Path(binary).resolve())
    if not os.path.isfile(binary):
        alt = Path(__file__).resolve().parent.parent / "build" / "record_permutation_lsh"
        if alt.is_file():
            binary = str(alt)
    if not os.path.isfile(binary):
        print(f"Error: binary not found: {args.binary}", file=sys.stderr)
        return 1

    lines = list_path.read_text(encoding="utf-8", errors="replace").splitlines()
    log_file = open(args.output, "a", encoding="utf-8") if args.output else None

    csv_path = Path(args.csv) if args.csv else list_path.parent / (list_path.stem + "_record_perm_results.csv")
    fieldnames = [
        "mtxname",
        "k",
        "bands",
        "hc_v",
        "cluster_size",
        "out_dir",
        "perm_path",
        "offsets_path",
        "maxmem_gb",
        "exit_code",
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
            if len(parts) < 3:
                print(f"Skip line {i}: need at least name k bands", file=sys.stderr)
                continue
            name, k, bands = parts[0], parts[1], parts[2]

            mtx = os.path.join(base_dir, name, f"{name}.mtx")
            perm_path = os.path.join(out_dir, f"{name}.perm")
            offsets_path = os.path.join(out_dir, f"{name}.offsets")

            row = {
                "mtxname": name,
                "k": k,
                "bands": bands,
                "hc_v": args.hc_v,
                "cluster_size": args.cluster_size,
                "out_dir": out_dir,
                "perm_path": perm_path,
                "offsets_path": offsets_path,
                "maxmem_gb": "",
                "exit_code": "",
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

            cmd = [
                binary,
                mtx,
                f"--k={k}",
                f"--bands={bands}",
                f"--hc_v={args.hc_v}",
                f"--cluster_size={args.cluster_size}",
                f"--out-dir={out_dir}",
                f"--precision={args.precision}",
            ]

            print(f"[{i}/{len(lines)}] Record {name} k={k} bands={bands} ...", flush=True)
            total += 1
            try:
                run_cmd = ["/usr/bin/time", "-v"] + cmd if args.record_maxmem else cmd
                result = subprocess.run(
                    run_cmd,
                    capture_output=True,
                    text=True,
                    timeout=None,
                )
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                row["exit_code"] = str(result.returncode)

                if log_file:
                    log_file.write(f"\n===== {name} k={k} bands={bands} =====\n")
                    log_file.write("CMD: " + " ".join(run_cmd) + "\n")
                    log_file.write(stdout)
                    if stderr:
                        log_file.write(stderr)
                    log_file.write(f"\nExit code: {result.returncode}\n")
                    log_file.flush()

                if args.record_maxmem:
                    m = re.search(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", stderr)
                    if m:
                        maxmem_kb = float(m.group(1))
                        row["maxmem_gb"] = f"{(maxmem_kb / (1024.0 * 1024.0)):.6f}"

                if result.returncode != 0:
                    failed += 1
                    row["error"] = (stderr.strip() or f"exit code {result.returncode}")[:500]
                    print(f"  FAIL: exit code {result.returncode}", file=sys.stderr)
                else:
                    # Verify outputs exist
                    ok = os.path.isfile(perm_path) and os.path.isfile(offsets_path)
                    if not ok:
                        failed += 1
                        row["error"] = "output files missing"
                        print("  FAIL: output files missing", file=sys.stderr)
                    elif args.record_maxmem and not row["maxmem_gb"]:
                        failed += 1
                        row["error"] = "maxmem_parse_failed"
                        print("  FAIL: maxmem_parse_failed", file=sys.stderr)
                    else:
                        print("  OK", flush=True)

            except Exception as e:
                failed += 1
                row["error"] = str(e)[:500]
                row["exit_code"] = ""
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

