from __future__ import annotations

from datasets import Audio, DatasetDict, load_dataset


def split_files(config: dict, split: str) -> str:
    # Mirrors the data_files of the official `<iso>_asr` config. Naming the
    # files explicitly avoids downloading the config's `unlabeled` split, which
    # is 20-35 GB per language and unused here.
    prefix = config["hf_dataset_config"].split("_")[0]
    return f"data/ASR/{prefix}/{prefix}-{split}-*"


def load_waxal(config: dict, splits: list[str] | None = None) -> DatasetDict:
    """Load only the requested labelled WAXAL splits, decoding audio at 16 kHz."""
    splits = splits or [config["train_split"], config["val_split"], config["test_split"]]
    dataset = load_dataset(config["hf_dataset_name"], data_files={split: split_files(config, split) for split in splits},
                           cache_dir=config.get("cache_dir"))
    if config.get("max_samples"):  # smoke tests only
        dataset = DatasetDict({name: split.select(range(min(len(split), config["max_samples"]))) for name, split in dataset.items()})
    return dataset.cast_column(config["audio_column"], Audio(sampling_rate=16000))
