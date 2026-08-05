from __future__ import annotations

from datasets import DatasetDict, concatenate_datasets, load_from_disk

from src.data.load_multilingual import load_multilingual


def load_multilingual_with_pseudolabels(config: dict) -> DatasetDict:
    """Load multilingual WAXAL data and augment the train split with pseudo-labels.

    Expects config to have a `pseudo_label_dir` key pointing to a dataset
    saved by src.pseudo_label (a flat Dataset, not a DatasetDict).
    """
    base = load_multilingual(config)
    pseudo_dir = config.get("pseudo_label_dir")
    if not pseudo_dir:
        return base
    pseudo = load_from_disk(pseudo_dir)
    # Drop pseudo_confidence column if present so schemas match
    if "pseudo_confidence" in pseudo.column_names:
        pseudo = pseudo.remove_columns(["pseudo_confidence"])
    # Align columns to the labeled train split
    train_cols = set(base[config["train_split"]].column_names)
    pseudo_cols = set(pseudo.column_names)
    for col in pseudo_cols - train_cols:
        pseudo = pseudo.remove_columns([col])
    augmented_train = concatenate_datasets([base[config["train_split"]], pseudo])
    return DatasetDict({
        config["train_split"]: augmented_train,
        config["val_split"]: base[config["val_split"]],
        config["test_split"]: base[config["test_split"]],
    })
