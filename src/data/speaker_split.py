from __future__ import annotations

import random

from datasets import DatasetDict


def apply_speaker_holdout(dataset_dict: DatasetDict, config: dict) -> DatasetDict:
    """Replace the validation split with a speaker-disjoint slice carved out of train.

    WAXAL's own provided train/validation/test splits share 60-90%+ of unique
    speakers across partitions (verified empirically against the HF dataset),
    so validation loss on the official split mostly measures how well a model
    has memorized already-heard voices, not generalization to unseen speakers
    -- the actual Phase 2 grading criterion for this competition. This instead
    holds out a fraction of unique speakers entirely, moving every utterance
    of theirs into validation, so early-stopping/eval_loss reflects genuine
    speaker generalization.

    Rows with no speaker_id (e.g. FLEURS augmentation rows, which are appended
    to train with speaker_id=None) are never selected for holdout and always
    stay in train, since they carry no speaker identity to hold out by.
    """
    train_split, val_split = config["train_split"], config["val_split"]
    fraction = config.get("speaker_holdout_fraction", 0.15)
    seed = config.get("seed", 42)

    train = dataset_dict[train_split]
    speaker_column = train["speaker_id"]
    unique_speakers = sorted({s for s in speaker_column if s is not None})
    rng = random.Random(seed)
    rng.shuffle(unique_speakers)
    n_holdout = max(1, round(len(unique_speakers) * fraction))
    holdout_speakers = set(unique_speakers[:n_holdout])

    holdout_indices = [i for i, s in enumerate(speaker_column) if s in holdout_speakers]
    keep_indices = [i for i, s in enumerate(speaker_column) if s not in holdout_speakers]

    result = dict(dataset_dict)
    result[val_split] = train.select(holdout_indices)
    result[train_split] = train.select(keep_indices)
    return DatasetDict(result)
