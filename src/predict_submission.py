"""Generate a Zindi submission CSV from fine-tuned per-language checkpoints.

The competition test set is exactly the WAXAL HF test split.  IDs are prefixed
with the language code (lin_, sna_, lug_), so we route each row to the correct
checkpoint without any language-detection step.

Usage:
    python -m src.predict_submission \
        --test-csv  zindi_data/Test.csv \
        --output    zindi_data/submission.csv \
        --lin-checkpoint /data/hf_cache/jgibson2/waxal/checkpoints/finetune_lingala_whisper_small/best_model \
        --sna-checkpoint /data/hf_cache/jgibson2/waxal/checkpoints/finetune_shona_whisper_small/best_model \
        --lug-checkpoint /data/hf_cache/jgibson2/waxal/checkpoints/finetune_luganda_whisper_small/best_model
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor,
                           WhisperForConditionalGeneration, WhisperProcessor)

from src.data.dataset import CTCCollator, WhisperCollator
from src.utils.metrics import normalize_text

try:
    from pyctcdecode import build_ctcdecoder
    _PYCTCDECODE = True
except ImportError:
    _PYCTCDECODE = False


def _derepeat(text: str, max_repeat: int = 1) -> str:
    """Truncate at the point where any n-gram (n=1..6) repeats more than
    max_repeat times consecutively — the signature of Whisper hallucination."""
    words = text.split()
    for n in range(1, 7):
        for i in range(len(words) - n * max_repeat):
            ngram = tuple(words[i:i + n])
            reps = 1
            j = i + n
            while j + n <= len(words) and tuple(words[j:j + n]) == ngram:
                reps += 1
                j += n
            if reps > max_repeat:
                return " ".join(words[:i + n * max_repeat])
    return text


_HF_CONFIGS = {"lin": "lin_asr", "sna": "sna_asr", "lug": "lug_asr"}
_CACHE_DIR = "/data/hf_cache/jgibson2/waxal/cache"
_LM_NAMES = {"lin": "lingala", "sna": "shona", "lug": "luganda"}
_LM_DIR = "/data/hf_cache/jgibson2/waxal/lm"


def _is_ctc(checkpoint: str) -> bool:
    import json
    cfg = json.load(open(Path(checkpoint) / "config.json"))
    return cfg.get("model_type") not in ("whisper",)


def _build_ctc_decoder(processor: Wav2Vec2Processor, lang: str | None = None, beam_width: int = 100):
    """Build a pyctcdecode beam search decoder from the processor vocab,
    fused with a KenLM n-gram LM (Wikipedia + WAXAL train transcripts,
    normalize_text-normalized) when one exists for this language."""
    vocab = processor.tokenizer.get_vocab()
    # pyctcdecode expects labels sorted by token id; blank must be index 0
    labels = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
    # Replace the tokenizer's pad token (used as CTC blank) with empty string at index 0
    blank_id = processor.tokenizer.pad_token_id
    labels[blank_id] = ""

    lm_name = _LM_NAMES.get(lang)
    lm_path = Path(_LM_DIR) / f"{lm_name}.binary" if lm_name else None
    corpus_path = Path(_LM_DIR) / f"{lm_name}_wiki.txt" if lm_name else None
    if lm_path and lm_path.exists():
        unigrams = None
        if corpus_path and corpus_path.exists():
            with open(corpus_path, encoding="utf-8") as f:
                unigrams = {word for line in f for word in line.split()}
        print(f"[{lang}] using KenLM fusion: {lm_path}")
        return build_ctcdecoder(labels, kenlm_model_path=str(lm_path), unigrams=unigrams), beam_width
    return build_ctcdecoder(labels), beam_width


def _load_model(checkpoint: str, device: torch.device):
    if _is_ctc(checkpoint):
        processor = Wav2Vec2Processor.from_pretrained(checkpoint)
        model = Wav2Vec2ForCTC.from_pretrained(checkpoint).to(device)
    else:
        processor = WhisperProcessor.from_pretrained(checkpoint)
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint).to(device)
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
    model.eval()
    return model, processor


def _predict(model, processor, dataset, batch_size: int, device: torch.device, lang: str | None = None) -> dict[str, str]:
    """Returns {id: prediction} for all examples in dataset."""
    ids = dataset["id"]
    is_ctc = isinstance(model, Wav2Vec2ForCTC)
    collate_fn = (
        CTCCollator(processor, "audio", "transcription")
        if is_ctc
        else WhisperCollator(processor, "audio", "transcription")
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    predictions = []
    ctc_decoder, beam_width = (_build_ctc_decoder(processor, lang) if is_ctc and _PYCTCDECODE else (None, 0))
    if ctc_decoder is not None and model.lm_head.out_features != len(processor.tokenizer.get_vocab()):
        # Some older mms_1b_finetune checkpoints were trained with a
        # head/vocab size mismatch (see load_mms_1b_finetune's docstring in
        # src/models/mms_wrapper.py) -- pyctcdecode requires an exact match,
        # so fall back to greedy rather than crash on those checkpoints.
        print(f"WARNING: model output dim ({model.lm_head.out_features}) != processor vocab "
              f"({len(processor.tokenizer.get_vocab())}); falling back to greedy decoding for this checkpoint.")
        ctc_decoder = None
    with torch.inference_mode():
        for batch in loader:
            if is_ctc:
                logits = model(batch["input_values"].to(device),
                               attention_mask=batch["attention_mask"].to(device)).logits
                if ctc_decoder is not None:
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
                    predictions.extend(ctc_decoder.decode_batch(None, log_probs, beam_width=beam_width))
                else:
                    pred_ids = torch.argmax(logits, dim=-1)
                    predictions.extend(processor.batch_decode(pred_ids))
            else:
                generated = model.generate(
                    batch["input_features"].to(device), do_sample=False, num_beams=1,
                    no_repeat_ngram_size=4,
                )
                predictions.extend(processor.batch_decode(generated, skip_special_tokens=True))
    return {id_: (_derepeat(normalize_text(pred)) or "_") for id_, pred in zip(ids, predictions)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="zindi_data/Test.csv")
    parser.add_argument("--output", default="zindi_data/submission.csv")
    parser.add_argument("--multilingual-checkpoint", default=None,
                        help="Single checkpoint for all languages (overrides per-language flags)")
    parser.add_argument("--lin-checkpoint", default=None)
    parser.add_argument("--sna-checkpoint", default=None)
    parser.add_argument("--lug-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not args.multilingual_checkpoint and not all(
        [args.lin_checkpoint, args.sna_checkpoint, args.lug_checkpoint]
    ):
        parser.error("Provide --multilingual-checkpoint or all three per-language checkpoints.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.test_csv) as f:
        ordered_ids = [r["ID"] for r in csv.DictReader(f)]

    if args.multilingual_checkpoint:
        checkpoints = {lang: args.multilingual_checkpoint for lang in _HF_CONFIGS}
    else:
        checkpoints = {"lin": args.lin_checkpoint, "sna": args.sna_checkpoint, "lug": args.lug_checkpoint}

    id_to_pred: dict[str, str] = {}
    loaded_checkpoint = None
    loaded_model, loaded_processor = None, None

    for lang, checkpoint in checkpoints.items():
        if checkpoint != loaded_checkpoint:
            del loaded_model  # free GPU memory
            print(f"[{lang}] loading checkpoint {checkpoint}")
            loaded_model, loaded_processor = _load_model(checkpoint, device)
            loaded_checkpoint = checkpoint
        ds = (
            load_dataset("google/WaxalNLP", _HF_CONFIGS[lang], split="test", cache_dir=_CACHE_DIR)
            .cast_column("audio", Audio(sampling_rate=16000))
        )
        print(f"[{lang}] running inference on {len(ds)} examples")
        id_to_pred.update(_predict(loaded_model, loaded_processor, ds, args.batch_size, device, lang))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Target"])
        for id_ in ordered_ids:
            writer.writerow([id_, id_to_pred[id_]])

    print(f"Wrote {len(ordered_ids)} rows to {args.output}")


if __name__ == "__main__":
    main()
