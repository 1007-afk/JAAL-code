
#!/usr/bin/env python3
"""
Replace speaker labels in scam conversation dataset.
Replaces "Scammer:" with "Caller:" in all .txt files.

Usage:
    python relabel_speaker.py \
        --source scam_10k \
        --dest   scam_10k_relabeled \
        [--find  "Scammer:"] \
        [--replace "Caller:"] \
        [--inplace]

    --inplace : edit files in-place inside --source (no copy, careful!)
    Without --inplace: copies full folder structure to --dest with replacements applied.
"""

import argparse
import shutil
from pathlib import Path


def relabel_file(src: Path, dst: Path, find: str, replace: str) -> int:
    """Read src, replace, write to dst. Returns number of replacements made."""
    text = src.read_text(encoding="utf-8", errors="ignore")
    new_text, count = text.replace(find, replace), text.count(find)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new_text, encoding="utf-8")
    return count


def main():
    parser = argparse.ArgumentParser(description="Replace speaker labels in scam dataset")
    parser.add_argument("--source",  required=True, help="Source folder")
    parser.add_argument("--dest",    default=None,  help="Destination folder (required unless --inplace)")
    parser.add_argument("--find",    default="Scammer:", help="String to find (default: 'Scammer:')")
    parser.add_argument("--replace", default="Caller:",  help="Replacement string (default: 'Caller:')")
    parser.add_argument("--inplace", action="store_true", help="Edit files in-place in source folder")
    args = parser.parse_args()

    if not args.inplace and not args.dest:
        print("ERROR: Provide --dest or use --inplace")
        return

    source = Path(args.source)
    dest   = Path(args.dest) if args.dest else source

    if not args.inplace and dest.exists():
        print(f"WARN: Destination '{dest}' already exists. Files will be overwritten.")

    all_txt = list(source.rglob("*.txt"))
    print(f"Found {len(all_txt)} .txt files in '{source}'")
    print(f"Replacing: '{args.find}' → '{args.replace}'")
    if args.inplace:
        print("Mode: IN-PLACE (modifying source files directly)")
    else:
        print(f"Mode: COPY to '{dest}'")
    print()

    total_files   = 0
    total_replacements = 0
    files_changed = 0

    for src_file in all_txt:
        rel      = src_file.relative_to(source)
        dst_file = dest / rel if not args.inplace else src_file

        count = relabel_file(src_file, dst_file, args.find, args.replace)
        total_files += 1
        total_replacements += count
        if count > 0:
            files_changed += 1

        if total_files % 500 == 0:
            print(f"  Processed {total_files}/{len(all_txt)} ...", end="\r")

    print(f"\nDone.")
    print(f"  Files processed : {total_files}")
    print(f"  Files changed   : {files_changed}")
    print(f"  Total replacements: {total_replacements}")
    if not args.inplace:
        print(f"  Output folder   : {dest.resolve()}")


if __name__ == "__main__":
    main()
