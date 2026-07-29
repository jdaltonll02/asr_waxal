"""Apply repetition post-processing to an existing submission CSV.

Usage:
    python -m src.postprocess_submission \
        --input  zindi_data/submission_multilingual.csv \
        --output zindi_data/submission_multilingual_clean.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.predict_submission import _derepeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fixed = changed = 0
    for row in rows:
        clean = _derepeat(row["Target"])
        if clean != row["Target"]:
            changed += 1
        row["Target"] = clean
        fixed += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {fixed} rows, fixed {changed} ({100*changed/fixed:.1f}%)")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
