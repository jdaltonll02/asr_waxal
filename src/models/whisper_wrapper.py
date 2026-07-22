from __future__ import annotations

from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_whisper(model_config: dict, language: str):
    processor = WhisperProcessor.from_pretrained(model_config["checkpoint"])
    model = WhisperForConditionalGeneration.from_pretrained(model_config["checkpoint"])
    # Greedy only: no forced language token is used because Lingala/Shona/Luganda
    # do not all have Whisper language tokens.
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    return model, processor
