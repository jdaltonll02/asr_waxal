from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.utils.metrics import normalize_text


def audio_array(example: dict, audio_column: str):
    return example[audio_column]["array"]


@dataclass
class WhisperCollator:
    processor: object
    audio_column: str
    text_column: str

    def __call__(self, features):
        # Whisper's encoder requires exactly 3000 mel frames (30s); the feature
        # extractor's default padding="max_length" pads/truncates to that fixed
        # length regardless of batch content. Do not override it with padding=True,
        # which pads only to the longest example in the batch and produces
        # under-length features whenever no clip in the batch reaches 30s.
        inputs = self.processor.feature_extractor(
            [audio_array(item, self.audio_column) for item in features], sampling_rate=16000,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer([item[self.text_column] for item in features], return_tensors="pt", padding=True).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"input_features": inputs.input_features, "labels": labels}


@dataclass
class CTCCollator:
    processor: object
    audio_column: str
    text_column: str

    def __call__(self, features):
        inputs = self.processor([audio_array(item, self.audio_column) for item in features], sampling_rate=16000,
                                return_tensors="pt", padding=True)
        # as_target_processor() is deprecated in transformers>=4.5x and its
        # merged-kwargs __call__ raises "got multiple values for keyword
        # argument 'padding'" here; call the tokenizer directly instead.
        labels = self.processor.tokenizer([normalize_text(item[self.text_column]) for item in features], return_tensors="pt", padding=True).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"input_values": inputs.input_values, "attention_mask": inputs.attention_mask, "labels": labels}


@dataclass
class AugmentedCTCCollator(CTCCollator):
    """CTCCollator that applies on-the-fly audio augmentation during training.

    A TrainingModeCallback (see train.py) toggles `self.training` before each
    train/eval phase so augmentation is skipped during evaluation.
    """
    aug_kwargs: dict = field(default_factory=dict)
    training: bool = True

    def __call__(self, features):
        if not self.training:
            return super().__call__(features)
        from src.data.augment import augment_audio
        augmented = [augment_audio(audio_array(item, self.audio_column), **self.aug_kwargs)
                     for item in features]
        patched = [{**item, self.audio_column: {**item[self.audio_column], "array": aug}}
                   for item, aug in zip(features, augmented)]
        return super().__call__(patched)
