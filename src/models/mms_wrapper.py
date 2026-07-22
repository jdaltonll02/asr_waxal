from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoFeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor


def build_vocab(train_dataset, text_column: str, output_dir: str) -> str:
    characters = sorted({char for text in train_dataset[text_column] for char in text.lower()})
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
                                            ctc_loss_reduction="mean")
    if model_config["freeze_encoder"]:
        for parameter in model.wav2vec2.parameters():
            parameter.requires_grad = False
    return model, processor


def load_mms_1b(model_config: dict, language_code: str):
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    processor = AutoProcessor.from_pretrained(model_config["checkpoint"])
    processor.tokenizer.set_target_lang(language_code)
    model = Wav2Vec2ForCTC.from_pretrained(model_config["checkpoint"])
    model.load_adapter(language_code)
    return model, processor
