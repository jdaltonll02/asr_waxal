from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoFeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

from src.utils.metrics import normalize_text


def build_vocab(train_dataset, text_column: str, output_dir: str) -> str:
    # Normalize before building vocab so punctuation/digits that carry no
    # acoustic signal don't become CTC targets. Raw transcriptions in
    # Lingala/Shona contain thousands of commas and periods which fragment
    # the output space and prevent the CTC head from learning.
    characters = sorted({char for text in train_dataset[text_column]
                         for char in normalize_text(text) if char != " "})
    vocab = {char: index for index, char in enumerate(characters)}
    vocab["|"] = len(vocab)  # word delimiter (replaces space at decode time)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    path = Path(output_dir) / "vocab.json"
    path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_mms_300m(model_config: dict, train_dataset, text_column: str, output_dir: str):
    vocab_path = build_vocab(train_dataset, text_column, output_dir)
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_config["checkpoint"])
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_config["checkpoint"], vocab_size=len(tokenizer),
        ignore_mismatched_sizes=True, pad_token_id=tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        # SpecAugment masking (Wav2Vec2Model._mask_hidden_states) is gated on
        # model.training, so this is inert for evaluate.py/cross_domain_eval.py,
        # which call model.eval() before inference. MMS's own base config
        # already enables apply_spec_augment with mask_time_prob=0.075 but
        # mask_feature_prob=0.0; only feature masking needs opting in here.
        apply_spec_augment=model_config.get("apply_spec_augment", True),
        mask_time_prob=model_config.get("mask_time_prob", 0.075),
        mask_feature_prob=model_config.get("mask_feature_prob", 0.0),
    )
    if model_config["freeze_encoder"]:
        # Freeze only the CNN feature extractor (~4.2M of ~315M wav2vec2
        # params), matching the standard wav2vec2 CTC fine-tuning recipe.
        # Freezing the full backbone (feature_projection + transformer
        # encoder too) left the from-scratch CTC head with no adaptable
        # signal to learn from: it collapsed to predicting one near-constant
        # output regardless of input audio, at both 1e-4 and 5e-4 learning
        # rates (see finetune_*_mms_300m runs prior to this fix).
        for parameter in model.wav2vec2.feature_extractor.parameters():
            parameter.requires_grad = False
    return model, processor


def load_mms_1b(model_config: dict, language_code: str):
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    processor = AutoProcessor.from_pretrained(model_config["checkpoint"])
    processor.tokenizer.set_target_lang(language_code)
    model = Wav2Vec2ForCTC.from_pretrained(model_config["checkpoint"])
    model.load_adapter(language_code)
    return model, processor


def load_mms_1b_finetune(model_config: dict, language_code: str, train_dataset, text_column: str, output_dir: str):
    """Load MMS-1B with its pre-trained language adapter, then replace the LM
    head with a fresh CTC head built from the training vocabulary.

    model.load_adapter() swaps in that adapter's own native lm_head, sized for
    MMS's own per-language vocabulary (confirmed empirically: 81 classes for
    Lingala, 65 for Shona, 79 for Luganda) -- it silently overwrites whatever
    vocab_size was requested when the model was constructed. It must run
    first; the custom head is rebuilt afterward to match our own training
    vocab. Getting this order backwards previously trained Lingala (64 vs 81)
    and Shona (47 vs 65) with a head-vs-label size mismatch throughout
    training -- Luganda (79 vs 79) only appeared fine because its vocab size
    happened to coincide with MMS's own.
    """
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor as W2V2Proc
    vocab_path = build_vocab(train_dataset, text_column, output_dir)
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_config["checkpoint"])
    processor = W2V2Proc(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = Wav2Vec2ForCTC.from_pretrained(
        model_config["checkpoint"],
        apply_spec_augment=model_config.get("apply_spec_augment", True),
        mask_time_prob=model_config.get("mask_time_prob", 0.075),
        mask_feature_prob=model_config.get("mask_feature_prob", 0.0),
    )
    model.load_adapter(language_code)

    vocab_size = len(tokenizer)
    model.lm_head = torch.nn.Linear(model.config.hidden_size, vocab_size)
    model.config.vocab_size = vocab_size
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.ctc_loss_reduction = "mean"

    if model_config.get("freeze_feature_encoder", True):
        for parameter in model.wav2vec2.feature_extractor.parameters():
            parameter.requires_grad = False
    return model, processor
