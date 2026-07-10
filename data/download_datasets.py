#!/usr/bin/env python3
"""
Download matrices listed in SparseOps testdatasets.txt / alldatasets.txt
from the SuiteSparse Matrix Collection via ssgetpy, or download the
precomputed SparseOps reordering archive from GitHub Releases.

Matrix download requires: pip install ssgetpy requests tqdm
Reordering download requires: curl, GNU tar, and zstd

Example:
  python download_datasets.py --source test
  python download_datasets.py --source all --format MM --out /path/to/dir
  python download_datasets.py --source test --dry-run
  python download_datasets.py --reordering
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


REORDERING_ARCHIVE_NAME = "sparseops-reordering-v1.tar.zst"
REORDERING_RELEASE_URL = (
    "https://github.com/lsl036/SparseOps/releases/download/"
    f"reordering-v1/{REORDERING_ARCHIVE_NAME}"
)
REORDERING_SHA256 = "854d9ca01c9ddb2b0f2f92664ee980331811afa39267513be215c9cf93b134de"


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download_reordering(out_dir: Path, dry_run: bool, skip_existing: bool) -> int:
    target_dir = out_dir / "reordering"
    if skip_existing and target_dir.is_dir() and any(target_dir.iterdir()):
        print(f"skip existing reordering data: {target_dir}")
        return 0

    print(f"reordering archive: {REORDERING_RELEASE_URL}")
    print(f"extract to: {target_dir}")
    if dry_run:
        return 0

    missing_tools = [name for name in ("curl", "tar", "zstd") if shutil.which(name) is None]
    if missing_tools:
        print(
            "error: reordering extraction requires: " + ", ".join(missing_tools),
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f".{REORDERING_ARCHIVE_NAME}.part"

    try:
        archive_ready = (
            archive_path.is_file() and _file_sha256(archive_path) == REORDERING_SHA256
        )
        if archive_ready:
            print(f"using verified partial/download cache: {archive_path}")
        else:
            print("downloading reordering archive (automatic retry/resume enabled)...", flush=True)
            subprocess.run(
                [
                    "curl",
                    "--location",
                    "--fail",
                    "--retry",
                    "20",
                    "--retry-delay",
                    "3",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "30",
                    "--continue-at",
                    "-",
                    "--output",
                    str(archive_path),
                    REORDERING_RELEASE_URL,
                ],
                check=True,
            )

        actual_sha256 = _file_sha256(archive_path)
        if actual_sha256 != REORDERING_SHA256:
            print(
                "error: reordering archive SHA256 mismatch\n"
                f"  expected: {REORDERING_SHA256}\n"
                f"  actual:   {actual_sha256}",
                file=sys.stderr,
            )
            archive_path.unlink(missing_ok=True)
            return 1

        subprocess.run(
            ["tar", "--zstd", "-xf", str(archive_path), "-C", str(out_dir)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: failed to download/extract reordering archive: {exc}", file=sys.stderr)
        if archive_path.exists():
            print(f"partial archive kept for resume: {archive_path}", file=sys.stderr)
        return 1

    archive_path.unlink(missing_ok=True)
    print(f"reordering data ready: {target_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download SuiteSparse matrices named in testdatasets.txt / alldatasets.txt, "
            "or the SparseOps reordering archive."
        )
    )
    parser.add_argument(
        "--reordering",
        action="store_true",
        help="Download and extract the precomputed reordering archive, then exit.",
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

    out_dir = args.out.resolve() if args.out else Path(__file__).resolve().parent
    if args.reordering:
        return _download_reordering(out_dir, args.dry_run, args.skip_existing)

    try:
        import ssgetpy
    except ImportError:
        print(
            "error: ssgetpy is not installed. Run: pip install ssgetpy requests tqdm",
            file=sys.stderr,
        )
        return 1

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
