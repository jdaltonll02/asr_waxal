from __future__ import annotations

from functools import lru_cache

from datasets import Audio, DatasetDict, load_dataset
from huggingface_hub import HfApi


@lru_cache(maxsize=None)
def official_data_files(dataset_name: str, config_name: str) -> dict[str, str]:
    """Split -> file pattern, as declared by the dataset card's config.

    WAXAL's file naming is not uniform (dga/dag/kpo use `train-*`, the rest
    `<iso>-train-*`), so the patterns come from the card instead of a guess.
    """
    configs = {config["config_name"]: config for config in HfApi().dataset_info(dataset_name).card_data.configs}
    return {entry["split"]: entry["path"] for entry in configs[config_name]["data_files"]}


def load_waxal(config: dict, splits: list[str] | None = None) -> DatasetDict:
    """Load only the requested labelled WAXAL splits, decoding audio at 16 kHz.

    Naming the split files explicitly avoids downloading the config's
    `unlabeled` split, which is 20-35 GB per language and unused here.
    """
    splits = splits or [config["train_split"], config["val_split"], config["test_split"]]
    patterns = official_data_files(config["hf_dataset_name"], config["hf_dataset_config"])
    dataset = load_dataset(config["hf_dataset_name"], data_files={split: patterns[split] for split in splits},
                           cache_dir=config.get("cache_dir"))
    if config.get("max_samples"):  # smoke tests only
        dataset = DatasetDict({name: split.select(range(min(len(split), config["max_samples"]))) for name, split in dataset.items()})
    return dataset.cast_column(config["audio_column"], Audio(sampling_rate=16000))
