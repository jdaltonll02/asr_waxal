from __future__ import annotations

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset


_FLEURS_AUDIO_COL = "audio"
_FLEURS_TEXT_COL = "transcription"


def load_waxal_with_fleurs(config: dict) -> DatasetDict:
    """Load WAXAL splits and augment the train split with FLEURS train data.

    Requires config keys: fleurs_dataset_name, fleurs_config, audio_column,
    text_column, cache_dir (optional).
    FLEURS columns are renamed to match WAXAL's audio/text column names and
    an 'id' column is added so downstream code that references 'id' doesn't break.
    """
    cache = config.get("cache_dir")
    waxal = load_dataset(
        config["hf_dataset_name"], config["hf_dataset_config"], cache_dir=cache,
    ).cast_column(config["audio_column"], Audio(sampling_rate=16000))

    fleurs = load_dataset(
        config["fleurs_dataset_name"], config["fleurs_config"],
        split="train", cache_dir=cache,
    ).cast_column(_FLEURS_AUDIO_COL, Audio(sampling_rate=16000))

    # Keep only the columns we need, rename to match WAXAL schema
    audio_col, text_col = config["audio_column"], config["text_column"]
    fleurs = fleurs.select_columns(["id", _FLEURS_AUDIO_COL, _FLEURS_TEXT_COL])
    if _FLEURS_AUDIO_COL != audio_col:
        fleurs = fleurs.rename_column(_FLEURS_AUDIO_COL, audio_col)
    if _FLEURS_TEXT_COL != text_col:
        fleurs = fleurs.rename_column(_FLEURS_TEXT_COL, text_col)
    # Add a synthetic id column so clean_filter and any id-referencing code works
    fleurs = fleurs.map(lambda ex, idx: {"id": f"fleurs_{idx}"}, with_indices=True)

    augmented_train = concatenate_datasets([waxal["train"], fleurs])

    return DatasetDict({
        config["train_split"]: augmented_train,
        config["val_split"]: waxal[config["val_split"]],
        config["test_split"]: waxal[config["test_split"]],
    })
