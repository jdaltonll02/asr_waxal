from __future__ import annotations

import io

import soundfile as sf
from datasets import Audio

DURATION_COLUMN = "duration_sec"


def _header_duration(audio: dict) -> float:
    """Duration from the encoded file header, without decoding/resampling samples."""
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    try:
        return float(sf.info(source).duration)
    except Exception:
        # Formats libsndfile cannot probe fall back to a full decode.
        decoded = Audio().decode_example(audio)
        return len(decoded["array"]) / float(decoded["sampling_rate"])


def add_durations(dataset, config: dict):
    """Attach a duration column, reading audio headers only (duration is sample-rate independent)."""
    if DURATION_COLUMN in dataset.column_names:
        return dataset
    encoded = dataset.cast_column(config["audio_column"], Audio(decode=False))
    durations = encoded.map(lambda audio: {DURATION_COLUMN: _header_duration(audio)},
                            input_columns=[config["audio_column"]], remove_columns=encoded.column_names,
                            num_proc=config.get("num_proc"), desc="Reading durations")[DURATION_COLUMN]
    return dataset.add_column(DURATION_COLUMN, durations)


def is_valid(duration: float, text: str, config: dict) -> bool:
    words = len(str(text).split())
    return duration >= config["min_duration_sec"] and words / duration <= config["max_words_per_sec"]


def clean_split(dataset, config: dict):
    # input_columns keeps the audio column undecoded during filtering.
    return dataset.filter(is_valid, input_columns=[DURATION_COLUMN, config["text_column"]],
                          fn_kwargs={"config": config}, num_proc=config.get("num_proc"))


def split_stats(dataset) -> dict[str, float | int]:
    return {"utterances": len(dataset), "hours": sum(dataset[DURATION_COLUMN]) / 3600}


def clean_dataset(dataset_dict, config: dict, logger=None):
    cleaned = {}
    stats = {}
    for split_name in dataset_dict:
        with_durations = add_durations(dataset_dict[split_name], config)
        before = split_stats(with_durations)
        cleaned[split_name] = clean_split(with_durations, config)
        after = split_stats(cleaned[split_name])
        stats[split_name] = {"before": before, "after": after}
        if logger:
            logger.info("%s: %d / %.2f h -> %d / %.2f h", split_name, before["utterances"], before["hours"], after["utterances"], after["hours"])
    return cleaned, stats
