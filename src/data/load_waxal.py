from __future__ import annotations

from datasets import Audio, DatasetDict, load_dataset


def load_waxal(config: dict) -> DatasetDict:
    """Load the official WAXAL ASR config, decoding audio at 16 kHz."""
    dataset = load_dataset(
        config["hf_dataset_name"], config["hf_dataset_config"],
        cache_dir=config.get("cache_dir"),
    )
    return dataset.cast_column(config["audio_column"], Audio(sampling_rate=16000))
