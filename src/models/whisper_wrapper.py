from __future__ import annotations

from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_whisper(model_config: dict, language: str):
    processor = WhisperProcessor.from_pretrained(model_config["checkpoint"])
    model = WhisperForConditionalGeneration.from_pretrained(
        model_config["checkpoint"],
        # SpecAugment is disabled by default in WhisperConfig; the masking
        # itself is gated on model.training internally (transformers'
        # WhisperEncoder._mask_input_features), so this is inert for
        # evaluate.py/zero_shot_eval.py/cross_domain_eval.py, which all call
        # model.eval() before inference.
        apply_spec_augment=model_config.get("apply_spec_augment", False),
        mask_time_prob=model_config.get("mask_time_prob", 0.05),
        mask_feature_prob=model_config.get("mask_feature_prob", 0.0),
    )
    # Greedy only: no forced language token is used because Lingala/Shona/Luganda
    # do not all have Whisper language tokens.
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    return model, processor
