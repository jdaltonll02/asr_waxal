from __future__ import annotations

from collections.abc import Callable


def duration_seconds(example: dict, audio_column: str) -> float:
    audio = example[audio_column]
    array = audio["array"]
    return len(array) / float(audio["sampling_rate"])


def is_valid(example: dict, config: dict) -> bool:
    duration = duration_seconds(example, config["audio_column"])
    words = len(str(example[config["text_column"]]).split())
    return duration >= config["min_duration_sec"] and words / duration <= config["max_words_per_sec"]


def clean_split(dataset, config: dict):
    return dataset.filter(is_valid, fn_kwargs={"config": config}, num_proc=config.get("num_proc"))


def split_stats(dataset, config: dict) -> dict[str, float | int]:
    durations = [duration_seconds(example, config["audio_column"]) for example in dataset]
    return {"utterances": len(dataset), "hours": sum(durations) / 3600}


def clean_dataset(dataset_dict, config: dict, logger=None):
    cleaned = {}
    stats = {}
    for split_name in (config["train_split"], config["val_split"], config["test_split"]):
        before = split_stats(dataset_dict[split_name], config)
        cleaned[split_name] = clean_split(dataset_dict[split_name], config)
        after = split_stats(cleaned[split_name], config)
        stats[split_name] = {"before": before, "after": after}
        if logger:
            logger.info("%s: %d / %.2f h -> %d / %.2f h", split_name, before["utterances"], before["hours"], after["utterances"], after["hours"])
    return cleaned, stats
