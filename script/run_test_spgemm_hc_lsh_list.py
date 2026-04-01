#!/usr/bin/env python3
"""
Run test_spgemm_hc_lsh for each line in a dataset list file.
Each line: dataset_name  k  bands  [optional rest ignored]
Command: test_spgemm_hc_lsh BASE/name/name.mtx BASE/name/name.mtx --k=k --bands=bands --hc_v=0 --kernel=3
OMP: OMP_PLACES=cores, OMP_PROC_BIND=spread. No timeout.
Writes CSV: mtxname,k,bands,r,genPairs_time,HC_time,Format_Conversion_time,LeSpGEMM_VLength_time,mixed_acc,phase_breakdown_avg,error
  With --print-bd, program prints [mixed_acc] phase breakdown avg ... (warmup breakdown is never printed).

Usage:
  # From build dir, list file at project root
  python3 ../script/run_test_spgemm_hc_lsh_list.py ../runable_datasets.txt

  # With results CSV (default: <list_file_stem>_results.csv)
  python3 ../script/run_test_spgemm_hc_lsh_list.py ../runable_datasets.txt -c run_results.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_run_output(text: str):
    """Extract times (ms) and [mixed_acc] from program stdout. Returns dict or None on failure."""
    out = {}
    m = re.search(r"genPairs time:\s*([\d.]+)\s*ms", text)
    if not m:
        return None
    out["genPairs_time"] = float(m.group(1))
    m = re.search(r"HC time:\s*([\d.]+)\s*ms", text)
    if not m:
        return None
    out["HC_time"] = float(m.group(1))
    m = re.search(r"Format Conversion time:\s*([\d.]+)\s*ms", text)
    if not m:
        return None
    out["Format_Conversion_time"] = float(m.group(1))
    m = re.search(r"Average time \(LeSpGEMM_VLength\):\s*([\d.]+)\s*ms", text)
    if not m:
        return None
    out["LeSpGEMM_VLength_time"] = float(m.group(1))
    # First [mixed_acc] line is dense clusters; optional second line is phase breakdown avg
    mixed_lines = re.findall(r"\[mixed_acc\][^\n]*", text)
    out["mixed_acc"] = mixed_lines[0].replace("[mixed_acc]", "").strip() if mixed_lines else ""
    out["phase_breakdown_avg"] = ""
    for line in mixed_lines:
        if "phase breakdown avg" in line:
            m_bd = re.search(r"phase breakdown avg over \d+ runs \(ms\):\s*(.+)", line)
            if m_bd:
                out["phase_breakdown_avg"] = m_bd.group(1).strip()
            break
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run test_spgemm_hc_lsh for each (name, k, bands) in a list file."
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
        default="./test_spgemm_hc_lsh",
        help="Path to test_spgemm_hc_lsh executable",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Append stdout/stderr of each run to FILE (optional)",
    )
    parser.add_argument(
        "-c", "--csv",
        metavar="FILE",
        help="Write results to CSV (default: <list_file_stem>_results.csv)",
    )
    parser.add_argument(
        "--print-bd",
        action="store_true",
        help="Pass --print_bd=1 to binary (kernel=3): print avg phase breakdown over iterations only",
    )
    args = parser.parse_args()

    list_path = Path(args.list_file)
    if not list_path.is_file():
        print(f"Error: list file not found: {list_path}", file=sys.stderr)
        sys.exit(1)

    base_dir = args.base_dir.rstrip("/")
    binary = args.binary
    if not os.path.isabs(binary):
        binary = str(Path(binary).resolve())
    if not os.path.isfile(binary):
        alt = Path(__file__).resolve().parent.parent / "build" / "test_spgemm_hc_lsh"
        if alt.is_file():
            binary = str(alt)
    if not os.path.isfile(binary):
        print(f"Error: binary not found: {args.binary}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["OMP_PLACES"] = "cores"
    env["OMP_PROC_BIND"] = "spread"

    lines = list_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    total = 0
    failed = 0
    log_file = open(args.output, "a", encoding="utf-8") if args.output else None

    csv_path = Path(args.csv) if args.csv else list_path.parent / (list_path.stem + "_results.csv")
    fieldnames = [
        "mtxname", "k", "bands", "r",
        "genPairs_time", "HC_time", "Format_Conversion_time", "LeSpGEMM_VLength_time",
        "mixed_acc", "phase_breakdown_avg", "error",
    ]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    try:
        for i, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) < 3:
                print(f"Skip line {i}: need at least name k bands", file=sys.stderr)
                continue
            name, k, bands = parts[0], parts[1], parts[2]
            try:
                ki, bi = int(k), int(bands)
                r = ki // bi if bi else 0
            except ValueError:
                r = ""
            mtx = os.path.join(base_dir, name, f"{name}.mtx")
            if not os.path.isfile(mtx):
                print(f"Skip {name}: matrix not found {mtx}", file=sys.stderr)
                failed += 1
                total += 1
                csv_writer.writerow({
                    "mtxname": name, "k": k, "bands": bands, "r": r,
                    "genPairs_time": "", "HC_time": "", "Format_Conversion_time": "", "LeSpGEMM_VLength_time": "",
                    "mixed_acc": "", "phase_breakdown_avg": "", "error": "matrix not found",
                })
                csv_file.flush()
                continue

            cmd = [
                binary,
                mtx,
                mtx,
                f"--k={k}",
                f"--bands={bands}",
                "--hc_v=0",
                "--kernel=3",
            ]
            if args.print_bd:
                cmd.append("--print_bd=1")
            print(f"[{i}/{len(lines)}] Run {name} k={k} bands={bands} ...", flush=True)
            total += 1
            row = {
                "mtxname": name, "k": k, "bands": bands, "r": r,
                "genPairs_time": "", "HC_time": "", "Format_Conversion_time": "", "LeSpGEMM_VLength_time": "",
                "mixed_acc": "", "phase_breakdown_avg": "", "error": "",
            }
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
                    log_file.write(f"\n===== {name} k={k} bands={bands} =====\n")
                    log_file.write(stdout)
                    if stderr:
                        log_file.write(stderr)
                    log_file.write(f"\nExit code: {result.returncode}\n")
                    log_file.flush()

                if result.returncode != 0:
                    failed += 1
                    row["error"] = stderr.strip() or f"exit code {result.returncode}"
                    print(f"  FAIL: exit code {result.returncode}", file=sys.stderr)
                    if not log_file and stderr:
                        print(stderr[:500], file=sys.stderr)
                else:
                    parsed = parse_run_output(stdout)
                    if parsed:
                        row["genPairs_time"] = parsed["genPairs_time"]
                        row["HC_time"] = parsed["HC_time"]
                        row["Format_Conversion_time"] = parsed["Format_Conversion_time"]
                        row["LeSpGEMM_VLength_time"] = parsed["LeSpGEMM_VLength_time"]
                        row["mixed_acc"] = parsed.get("mixed_acc", "")
                        row["phase_breakdown_avg"] = parsed.get("phase_breakdown_avg", "")
                        print(f"  OK", flush=True)
                    else:
                        row["error"] = "parse_failed"
                        failed += 1
                        print(f"  FAIL: parse_failed", file=sys.stderr)
            except Exception as e:
                failed += 1
                row["error"] = str(e)
                print(f"  FAIL: {e}", file=sys.stderr)
                if log_file:
                    log_file.write(f"\nException: {e}\n")
                    log_file.flush()

            csv_writer.writerow(row)
            csv_file.flush()

        print(f"\nDone: {total - failed}/{total} passed, {failed} failed. CSV: {csv_path}", file=sys.stderr)
    finally:
        if log_file:
            log_file.close()
        csv_file.close()

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
