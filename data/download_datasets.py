#!/usr/bin/env python3
"""
Download matrices listed in SparseOps testdatasets.txt / alldatasets.txt
from the SuiteSparse Matrix Collection via ssgetpy.

Requires: pip install ssgetpy requests tqdm

Example:
  python download_datasets.py --source test
  python download_datasets.py --source all --format MM --out /path/to/dir
  python download_datasets.py --source test --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_list_path(source: str) -> Path:
    root = _repo_root()
    if source == "test":
        return root / "testdatasets.txt"
    if source == "all":
        return root / "alldatasets.txt"
    raise ValueError(f"unknown source: {source}")


def _read_matrix_names(paths: list[Path]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            print(f"warning: list file not found, skip: {path}", file=sys.stderr)
            continue
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s in seen:
                    continue
                seen.add(s)
                names.append(s)
    return names


def _find_matrix(ssgetpy, spec: str):
    """
    Return a single Matrix for spec.
    spec is either 'name' or 'Group/name' (SuiteSparse id).
    """
    if "/" in spec:
        found = ssgetpy.search(spec, limit=50)
        group, _, name = spec.partition("/")
        exact = [m for m in found if m.group == group and m.name == name]
    else:
        found = ssgetpy.search(name=spec, limit=500)
        exact = [m for m in found if m.name == spec]

    if len(exact) == 1:
        return exact[0]
    if len(exact) == 0:
        hint = (
            f"no exact match for {spec!r}; try 'Group/{spec}' as in sparse.tamu.edu"
            if "/" not in spec
            else "no exact match for group/name"
        )
        raise LookupError(hint)
    opts = ", ".join(f"{m.group}/{m.name}" for m in exact[:10])
    more = " ..." if len(exact) > 10 else ""
    raise LookupError(
        f"ambiguous name {spec!r}: {len(exact)} matrices ({opts}{more}); "
        "use Group/name in the list file"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download SuiteSparse matrices named in testdatasets.txt / alldatasets.txt."
    )
    parser.add_argument(
        "--source",
        choices=("test", "all", "both"),
        default="test",
        help="Which name list to use (default: test).",
    )
    parser.add_argument(
        "--format",
        choices=("MM", "MAT", "RB"),
        default="MM",
        help="Archive format (default: MM Matrix Market).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Download directory (default: this script's directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve names only; do not download.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if extracted matrix folder/file already exists.",
    )
    args = parser.parse_args()

    try:
        import ssgetpy
    except ImportError:
        print(
            "error: ssgetpy is not installed. Run: pip install ssgetpy requests tqdm",
            file=sys.stderr,
        )
        return 1

    out_dir = args.out.resolve() if args.out else Path(__file__).resolve().parent

    if args.source == "both":
        list_paths = [_default_list_path("test"), _default_list_path("all")]
    else:
        list_paths = [_default_list_path(args.source)]

    specs = _read_matrix_names(list_paths)
    if not specs:
        print("error: no matrix names found in list file(s)", file=sys.stderr)
        return 1

    ok, failed = 0, 0
    for i, spec in enumerate(specs, 1):
        try:
            m = _find_matrix(ssgetpy, spec)
        except LookupError as e:
            print(f"[{i}/{len(specs)}] FAIL {spec!r}: {e}", file=sys.stderr)
            failed += 1
            continue

        ident = f"{m.group}/{m.name}"
        if args.dry_run:
            print(f"[{i}/{len(specs)}] would download {ident} ({m.rows}x{m.cols}, nnz={m.nnz})")
            ok += 1
            continue

        if args.skip_existing:
            localpath, _ = m.localpath(
                format=args.format, destpath=str(out_dir), extract=True
            )
            if os.path.exists(localpath):
                print(f"[{i}/{len(specs)}] skip existing {ident} -> {localpath}")
                ok += 1
                continue

        print(f"[{i}/{len(specs)}] downloading {ident} ...")
        m.download(format=args.format, destpath=str(out_dir), extract=True)
        print(f"[{i}/{len(specs)}] done {ident}")
        ok += 1

    print(f"finished: {ok} ok, {failed} failed, total {len(specs)}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
