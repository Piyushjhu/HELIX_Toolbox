#!/usr/bin/env python3
"""Normalize Finder-renamed PDV CSV filenames.

Preview changes first; add --apply to rename files.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def cleaned_name(name: str) -> str:
    """Remove a Finder duplicate suffix such as ' (1)' and trailing spaces."""
    return re.sub(r"\s+\(1\)$", "", name).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the PDV files")
    parser.add_argument("--apply", action="store_true", help="Perform the renames (default: preview only)")
    args = parser.parse_args()

    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"Not a directory: {directory}")

    changes = [(path, path.with_name(cleaned_name(path.name)))
               for path in sorted(directory.iterdir())
               if path.is_file() and cleaned_name(path.name) != path.name]

    if not changes:
        print("No filenames need normalization.")
        return

    for source, target in changes:
        print(f"{source.name!r} -> {target.name!r}")

    conflicts = [target for source, target in changes if target.exists() and target != source]
    if conflicts:
        print("\nNothing renamed: these target filenames already exist:")
        for target in conflicts:
            print(f"  {target.name!r}")
        raise SystemExit(1)

    if not args.apply:
        print(f"\nPreview only: {len(changes)} file(s) would be renamed. Re-run with --apply to proceed.")
        return

    for source, target in changes:
        source.rename(target)
    print(f"Renamed {len(changes)} file(s).")


if __name__ == "__main__":
    main()
