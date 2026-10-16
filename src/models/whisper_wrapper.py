from __future__ import annotations

from transformers import WhisperForConditionalGeneration, WhisperProcessor


def configure_decoding(model, language: str | None = None):
    """Pin the decoder prompt so it matches how the model is used.

    Clearing only generation_config.forced_decoder_ids is not enough: generate()
    then falls back to model.config.forced_decoder_ids, i.e. <|en|><|transcribe|>,
    and decodes every utterance as English.

    language=None (fine-tuned): prompt <|startoftranscript|><|notimestamps|>, the
    prefix used for training labels; language detection is disabled because most
    WAXAL languages have no Whisper token. language="amharic" etc. (zero-shot):
    prompt <|startoftranscript|><|am|><|transcribe|><|notimestamps|>.
    """
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    if language is None:
        if hasattr(model.generation_config, "lang_to_id"):
            del model.generation_config.lang_to_id
    else:
        model.generation_config.language = language
        model.generation_config.task = "transcribe"
    return model


def load_whisper(model_config: dict, language: str | None = None):
    """Load a pretrained Whisper; pass language only for zero-shot decoding."""
    processor = WhisperProcessor.from_pretrained(model_config["checkpoint"])
    model = WhisperForConditionalGeneration.from_pretrained(model_config["checkpoint"])
    return configure_decoding(model, language), processor
