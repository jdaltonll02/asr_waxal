from __future__ import annotations

from dataclasses import dataclass

import torch


def audio_array(example: dict, audio_column: str):
    return example[audio_column]["array"]


def ctc_text(text: str) -> str:
    """CTC target text; must match the normalization used to build the MMS vocabulary."""
    return str(text).lower()


def mask_padding(tokenized, pad_token_id: int) -> torch.Tensor:
    # Mask by attention_mask, not pad_token_id: Whisper's pad token is its EOS
    # token, so masking by id would also drop the EOS label and the model would
    # never learn to stop (unbounded repetition at generation time).
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is None:
        return tokenized.input_ids.masked_fill(tokenized.input_ids == pad_token_id, -100)
    return tokenized.input_ids.masked_fill(attention_mask.ne(1), -100)


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
        tokenized = self.processor.tokenizer([item[self.text_column] for item in features], return_tensors="pt", padding=True)
        labels = mask_padding(tokenized, self.processor.tokenizer.pad_token_id)
        # The model prepends <|startoftranscript|> itself when shifting labels
        # right; drop the tokenizer's copy so it is not predicted twice.
        start = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if (labels[:, 0] == start).all():
            labels = labels[:, 1:]
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
        tokenized = self.processor.tokenizer([ctc_text(item[self.text_column]) for item in features], return_tensors="pt", padding=True)
        return {"input_values": inputs.input_values, "attention_mask": inputs.attention_mask, "labels": mask_padding(tokenized, self.processor.tokenizer.pad_token_id)}
