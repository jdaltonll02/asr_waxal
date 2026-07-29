from __future__ import annotations

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset


def load_multilingual(config: dict) -> DatasetDict:
    """Load and concatenate WAXAL ASR configs for all configured languages."""
    splits = [config["train_split"], config["val_split"], config["test_split"]]
    combined = {}
    for split in splits:
        parts = [
            load_dataset(config["hf_dataset_name"], cfg, split=split, cache_dir=config.get("cache_dir"))
            for cfg in config["hf_dataset_configs"]
        ]
        combined[split] = concatenate_datasets(parts).cast_column(
            config["audio_column"], Audio(sampling_rate=16000)
        )
    return DatasetDict(combined)
