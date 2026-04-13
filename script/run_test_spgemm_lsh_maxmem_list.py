#!/usr/bin/env python3
"""
Batch-run test_spgemm_lsh with /usr/bin/time -v and record max memory usage.

Input list file format (whitespace-separated):
  mtx_name   k   bands   [optional rest ignored]

Output format (two columns):
  mtx_name MaxMem(GB)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_max_rss_kb(stderr_text: str):
    m = re.search(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", stderr_text)
    return int(m.group(1)) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run test_spgemm_lsh and record max RSS memory (GB) for each dataset."
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
        default="/data/linshengle_data/SpGEMM-Reordering/lsh_order",
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
        help="Output file path (default: <list_file_stem>_maxmem.txt)",
    )
    args = parser.parse_args()

    list_path = Path(args.list_file)
    if not list_path.is_file():
        print(f"Error: list file not found: {list_path}", file=sys.stderr)
        return 1

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

    output_path = Path(args.output) if args.output else list_path.parent / (list_path.stem + "_maxmem.txt")
    lines = list_path.read_text(encoding="utf-8", errors="replace").splitlines()

    env = os.environ.copy()
    env["OMP_PLACES"] = "cores"
    env["OMP_PROC_BIND"] = "spread"

    total = 0
    failed = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write("mtx_name MaxMem(GB)\n")
        fout.flush()

        for i, line in enumerate(lines, 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if not parts:
                continue

            name = parts[0]
            mtx = os.path.join(args.base_dir.rstrip("/"), name, f"{name}.mtx")
            perm_path = os.path.join(args.lsh_order_dir, f"{name}.perm")
            offsets_path = os.path.join(args.lsh_order_dir, f"{name}.offsets")
            total += 1

            if not os.path.isfile(mtx):
                failed += 1
                print(f"[{i}/{len(lines)}] Skip {name}: matrix not found {mtx}", file=sys.stderr)
                fout.write(f"{name} NaN\n")
                fout.flush()
                continue
            if not os.path.isfile(perm_path) or not os.path.isfile(offsets_path):
                failed += 1
                print(
                    f"[{i}/{len(lines)}] Skip {name}: missing {perm_path} or {offsets_path}",
                    file=sys.stderr,
                )
                fout.write(f"{name} NaN\n")
                fout.flush()
                continue

            cmd = [
                binary,
                mtx,
                mtx,
                f"--lsh-order-dir={args.lsh_order_dir}",
                f"--iterations={args.iterations}",
                f"--l2_fraction={args.l2_fraction}",
                f"--precision={args.precision}",
                f"--sort={args.sort}",
            ]
            if args.threads:
                cmd.append(f"--threads={args.threads}")

            timed_cmd = ["/usr/bin/time", "-v"] + cmd
            print(f"[{i}/{len(lines)}] Measure MaxMem {name} ...", flush=True)
            try:
                result = subprocess.run(
                    timed_cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=None,
                )
            except Exception as e:
                failed += 1
                print(f"  FAIL: {e}", file=sys.stderr)
                fout.write(f"{name} NaN\n")
                fout.flush()
                continue

            rss_kb = parse_max_rss_kb(result.stderr or "")
            if result.returncode != 0 or rss_kb is None:
                failed += 1
                print(f"  FAIL: exit={result.returncode}, rss_parse={rss_kb is not None}", file=sys.stderr)
                fout.write(f"{name} NaN\n")
            else:
                max_mem_gb = rss_kb / (1024.0 * 1024.0)
                fout.write(f"{name} {max_mem_gb:.6f}\n")
                print(f"  OK: {max_mem_gb:.6f} GB", flush=True)
            fout.flush()

    print(
        f"\nDone: {total - failed}/{total} passed, {failed} failed. Output: {output_path}",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

