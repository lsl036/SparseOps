#!/usr/bin/env python3
"""
Parse run_test_spgemm_hc / run_test_hc_AAt output and produce a table:
[Dataset, FLOPs, Kernel, mixed_acc, Average time, Average GFLOPS]
"""

import re
import sys
import csv
from pathlib import Path


def parse_output(path: str) -> list[dict]:
    """Parse log file and return list of dicts with keys:
    Dataset, FLOPs, Kernel, mixed_acc, Average time, Average GFLOPS
    """
    results = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Split by dataset blocks
    blocks = re.split(r"\n===== Dataset: (.+?) =====\n", text)
    # blocks[0] is header before first dataset; then (name1, block1), (name2, block2), ...
    if len(blocks) < 2:
        return results

    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        block = blocks[i + 1] if i + 1 < len(blocks) else ""

        row = {
            "Dataset": name,
            "FLOPs": "",
            "Kernel": "",
            "mixed_acc": "",
            "Average time": "",
            "Average GFLOPS": "",
        }

        m = re.search(r"FLOPs:\s*(\d+)", block)
        if m:
            row["FLOPs"] = m.group(1)

        m = re.search(r"Kernel:\s*(\d+)", block)
        if m:
            row["Kernel"] = m.group(1)

        m = re.search(r"\[mixed_acc\]\s*(.+)", block)
        if m:
            row["mixed_acc"] = m.group(1).strip()

        m = re.search(r"Average time:\s*([\d.]+)\s*ms", block)
        if m:
            row["Average time"] = m.group(1)

        m = re.search(r"Average GFLOPS:\s*([\d.]+)", block)
        if m:
            row["Average GFLOPS"] = m.group(1)

        results.append(row)

    return results


def main():
    script_dir = Path(__file__).resolve().parent
    if len(sys.argv) < 2:
        inp = script_dir / "run_test_hc_AAt.out"
        out_path = script_dir / "run_test_hc_AAt_results.csv"
    else:
        inp = Path(sys.argv[1])
        out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.parent / (inp.stem + "_results.csv")

    results = parse_output(str(inp))
    if not results:
        print("No dataset blocks found.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["Dataset", "FLOPs", "Kernel", "mixed_acc", "Average time", "Average GFLOPS"]
    with open(str(out_path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"Wrote {len(results)} rows to {out_path}")
    # Pretty print first few
    for row in results[:5]:
        print(row)
    if len(results) > 5:
        print("...")


if __name__ == "__main__":
    main()
