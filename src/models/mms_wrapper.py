from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoFeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

from src.data.dataset import ctc_text


def build_vocab(train_dataset, text_column: str, output_dir: str) -> str:
    characters = sorted({char for text in train_dataset[text_column] for char in ctc_text(text)})
    vocab = {char: index for index, char in enumerate(characters)}
    vocab["|"] = vocab.pop(" ", len(vocab))
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
    model = Wav2Vec2ForCTC.from_pretrained(model_config["checkpoint"], vocab_size=len(tokenizer),
                                            ignore_mismatched_sizes=True, pad_token_id=tokenizer.pad_token_id,
                                            ctc_loss_reduction="mean",
                                            # fp16 CTC on T4s can hit inf losses on rare
                                            # unalignable (audio shorter than label) batches.
                                            ctc_zero_infinity=True)
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
