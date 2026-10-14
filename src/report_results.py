"""Render collected WAXAL metrics into the paper's language × model table shape."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ("omnilingual_1b", "mms_1b", "whisper_large_v3", "mms_300m", "whisper_small", "whisper_tiny")
LANGUAGES = ("acholi", "akan", "amharic", "dagaare", "dagbani", "ewe", "fula", "ikposo", "lingala", "luganda", "malagasy", "masaaba", "nyankole", "oromo", "shona", "sidama", "soga", "tigrinya", "wolaytta")


def metric_cell(path: Path) -> str:
    if not path.exists():
        return "—"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") == "N/A":
        return "N/A"
    return f"{result['wer_percent']:.1f} / {result['cer_percent']:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="/kaggle/working/outputs")
    parser.add_argument("--output", default="/kaggle/working/outputs/waxal_results.md")
    args = parser.parse_args()
    root = Path(args.outputs_dir)
    lines = ["| Language | Omni. (ZS) | MMS-1B (ZS) | Whisper L (ZS) | MMS-300M (FT) | Whisp. S (FT) | Whisp. T (FT) |",
             "|---|---|---|---|---|---|---|"]
    for language in LANGUAGES:
        cells = []
        for model in MODELS:
            stage = "zero_shot" if model in MODELS[:3] else "finetune"
            cells.append(metric_cell(root / f"{stage}_{language}_{model}" / "metrics.json"))
        lines.append("| " + language.title() + " | " + " | ".join(cells) + " |")
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
