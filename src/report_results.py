"""Render collected metrics as the paper's tables, next to the paper's own numbers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ("omnilingual_1b", "mms_1b", "whisper_large_v3", "mms_300m", "whisper_small", "whisper_tiny")
FINETUNED = MODELS[3:]
HEADERS = ("Omni. (ZS)", "MMS-1B (ZS)", "Whisper L (ZS)", "MMS-300M (FT)", "Whisp. S (FT)", "Whisp. T (FT)")
LANGUAGES = ("acholi", "akan", "amharic", "dagaare", "dagbani", "ewe", "fula", "ikposo", "lingala", "luganda", "malagasy", "masaaba", "nyankole", "oromo", "shona", "sidama", "soga", "tigrinya", "wolaytta")
FLEURS_LANGUAGES = ("amharic", "fula", "lingala", "luganda", "oromo", "shona")

# Paper Table 3 (WER %), in MODELS order; None = N/A.
PAPER_WER = {
    "acholi": (75.8, 80.5, None, 42.3, 42.3, 57.7), "akan": (59.9, 99.0, None, 34.2, 31.7, 37.9),
    "amharic": (51.1, 100.7, 254.3, 37.8, 33.6, 41.3), "dagaare": (70.4, 76.1, None, 34.9, 34.0, 37.3),
    "dagbani": (79.2, 82.0, None, 35.0, 34.0, 39.5), "ewe": (79.3, 88.7, None, 31.3, 32.3, 35.5),
    "fula": (60.6, 63.6, None, 40.6, 42.6, 35.5), "ikposo": (93.3, 103.1, None, 75.3, 77.5, 80.9),
    "lingala": (45.3, 44.7, 93.0, 42.6, 42.7, 49.0), "luganda": (42.9, 32.1, None, 16.9, 21.6, 33.8),
    "malagasy": (40.4, 55.5, 132.0, 12.8, 13.1, 17.7), "masaaba": (71.9, 96.7, None, 49.5, 75.5, 59.6),
    "nyankole": (59.9, 78.3, None, 38.6, 44.7, 46.7), "oromo": (46.0, 26.6, None, 26.9, 25.2, 29.3),
    "shona": (35.6, 36.9, 112.4, 25.0, 26.9, 31.4), "sidama": (75.6, 84.9, None, 35.6, 30.1, 34.4),
    "soga": (68.6, 70.4, None, 47.2, 57.1, 69.0), "tigrinya": (82.4, 109.0, None, 57.1, 53.5, 60.3),
    "wolaytta": (95.7, 90.6, None, 38.8, 39.5, 42.6),
}
# Paper Table 4 (CER %, fine-tuned models), in FINETUNED order.
PAPER_CER = {
    "acholi": (17.1, 19.2, 33.6), "akan": (10.9, 11.3, 15.7), "amharic": (13.2, 12.9, 16.8), "dagaare": (13.6, 14.3, 16.1),
    "dagbani": (11.2, 11.9, 15.6), "ewe": (9.6, 10.6, 12.6), "fula": (10.4, 12.0, 12.6), "ikposo": (24.4, 26.5, 28.4),
    "lingala": (18.9, 21.7, 30.9), "luganda": (3.4, 5.5, 12.8), "malagasy": (2.8, 3.5, 5.9), "masaaba": (11.0, 36.5, 42.8),
    "nyankole": (9.5, 16.7, 16.0), "oromo": (6.6, 6.8, 8.9), "shona": (4.3, 5.1, 8.9), "sidama": (8.2, 7.6, 9.5),
    "soga": (9.5, 16.6, 28.0), "tigrinya": (37.2, 38.7, 43.2), "wolaytta": (12.3, 13.2, 14.3),
}
# Paper Table 2 (FLEURS WER %, fine-tuned models), in FINETUNED order.
PAPER_FLEURS_WER = {
    "amharic": (39.5, 36.4, 45.4), "fula": (56.5, 66.8, 84.6), "lingala": (38.3, 39.5, 63.9),
    "luganda": (57.8, 66.1, 72.3), "oromo": (70.1, 65.5, 74.6), "shona": (39.3, 51.2, 69.7),
}


def load_metrics(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def comparison(ours: dict | None, key: str, paper: float | None) -> str:
    """'ours (paper, ±delta)', or what is known when one side is missing."""
    if ours is not None and ours.get("status") == "N/A":
        return "N/A"
    paper_text = "N/A" if paper is None else f"{paper:.1f}"
    if ours is None:
        return f"— ({paper_text})"
    value = ours[key]
    if paper is None:
        return f"{value:.1f} (N/A)"
    return f"{value:.1f} ({paper:.1f}, {value - paper:+.1f})"


def table(title: str, headers, rows) -> list[str]:
    return [f"## {title}", "", "| Language | " + " | ".join(headers) + " |", "|---" * (len(headers) + 1) + "|",
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def mean(values) -> str:
    values = [value for value in values if value is not None]
    return f"{sum(values) / len(values):.1f} (n={len(values)})" if values else "—"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="/kaggle/working/outputs")
    parser.add_argument("--output", default="/kaggle/working/outputs/waxal_results.md")
    args = parser.parse_args()
    root = Path(args.outputs_dir)

    def metrics(stage, language, model):
        return load_metrics(root / f"{stage}_{language}_{model}" / "metrics.json")

    lines = ["# WAXAL-NET reproduction", "", "Cells: ours (paper, ours - paper). — = not run yet.", ""]
    wer_rows, ours_by_model = [], {model: [] for model in MODELS}
    for language in LANGUAGES:
        cells = []
        for index, model in enumerate(MODELS):
            result = metrics("zero_shot" if model not in FINETUNED else "finetune", language, model)
            cells.append(comparison(result, "wer_percent", PAPER_WER[language][index]))
            if result and "wer_percent" in result:
                ours_by_model[model].append(result["wer_percent"])
        wer_rows.append([language.title(), *cells])
    wer_rows.append(["**Mean (ours)**", *(mean(ours_by_model[model]) for model in MODELS)])
    lines += table("WAXAL test WER % (paper Table 3)", HEADERS, wer_rows)

    cer_rows = [[language.title(), *(comparison(metrics("finetune", language, model), "cer_percent", PAPER_CER[language][index])
                                     for index, model in enumerate(FINETUNED))] for language in LANGUAGES]
    lines += table("WAXAL test CER % (paper Table 4)", HEADERS[3:], cer_rows)

    fleurs_rows = [[language.title(), *(comparison(metrics("cross_domain", language, model), "wer_percent", PAPER_FLEURS_WER[language][index])
                                        for index, model in enumerate(FINETUNED))] for language in FLEURS_LANGUAGES]
    lines += table("FLEURS WER %, fine-tuned models (paper Table 2)", HEADERS[3:], fleurs_rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
