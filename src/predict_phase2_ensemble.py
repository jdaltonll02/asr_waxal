"""Generate a Zindi Phase 2 submission by ensembling two multilingual MMS-300M
checkpoints. For each clip, both models run and the more confident one's
decoded hypothesis (mean max-softmax probability) wins -- same confidence
metric used earlier in this project for language routing, applied here
across model variants instead of languages.

Usage:
    python -m src.predict_phase2_ensemble \
        --test-csv  zindi_data/Test_phase2.csv \
        --audio-dir zindi_data/phase2_audio/audio \
        --checkpoint-a /data/hf_cache/jgibson2/waxal/checkpoints/finetune_multilingual_mms_300m_largebatch/best_model \
        --checkpoint-b /data/hf_cache/jgibson2/waxal/checkpoints/finetune_multilingual_mms_300m_augmented/best_model \
        --output    zindi_data/submission_phase2_ensemble.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.predict_submission import _derepeat
from src.utils.metrics import normalize_text

from pyctcdecode import build_ctcdecoder

_LM_PATH = Path("/data/hf_cache/jgibson2/waxal/lm/multilingual.binary")
_LM_CORPUS = Path("/data/hf_cache/jgibson2/waxal/lm/multilingual.txt")
_LM_ALPHA, _LM_BETA = 0.0, 0.5


def _load_audio(path: Path) -> np.ndarray:
    array, sr = sf.read(str(path))
    if sr != 16000:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array.astype(np.float32)


def _build_decoder(processor: Wav2Vec2Processor):
    vocab = processor.tokenizer.get_vocab()
    labels = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
    labels[processor.tokenizer.pad_token_id] = ""
    seen: set[str] = set()
    for i, lbl in enumerate(labels):
        if lbl in seen:
            labels[i] = f"<dup{i}>"
        seen.add(lbl)
    unigrams = {word for line in open(_LM_CORPUS, encoding="utf-8") for word in line.split()}
    return build_ctcdecoder(labels, kenlm_model_path=str(_LM_PATH), unigrams=unigrams, alpha=_LM_ALPHA, beta=_LM_BETA)


def _load_model(ckpt: str, device: torch.device):
    processor = Wav2Vec2Processor.from_pretrained(ckpt)
    model = Wav2Vec2ForCTC.from_pretrained(ckpt).to(device).eval()
    decoder = _build_decoder(processor)
    return processor, model, decoder


def _confidence_and_logits(model, processor, array: np.ndarray, device: torch.device):
    inputs = processor([array], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).logits
    conf = torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item()
    return conf, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="zindi_data/Test_phase2.csv")
    parser.add_argument("--audio-dir", default="zindi_data/phase2_audio/audio")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--output", default="zindi_data/submission_phase2_ensemble.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_dir = Path(args.audio_dir)

    with open(args.test_csv) as f:
        ordered_ids = [r["ID"] for r in csv.DictReader(f)]

    print(f"Loading checkpoint A: {args.checkpoint_a}")
    proc_a, model_a, dec_a = _load_model(args.checkpoint_a, device)
    print(f"Loading checkpoint B: {args.checkpoint_b}")
    proc_b, model_b, dec_b = _load_model(args.checkpoint_b, device)

    predictions: dict[str, str] = {}
    a_wins = 0
    for n, id_ in enumerate(ordered_ids):
        array = _load_audio(audio_dir / f"{id_}.wav")
        conf_a, logits_a = _confidence_and_logits(model_a, proc_a, array, device)
        conf_b, logits_b = _confidence_and_logits(model_b, proc_b, array, device)
        if conf_a >= conf_b:
            a_wins += 1
            log_probs = torch.nn.functional.log_softmax(logits_a, dim=-1).cpu().numpy()
            pred = dec_a.decode_batch(None, log_probs, beam_width=100)[0]
        else:
            log_probs = torch.nn.functional.log_softmax(logits_b, dim=-1).cpu().numpy()
            pred = dec_b.decode_batch(None, log_probs, beam_width=100)[0]
        predictions[id_] = _derepeat(normalize_text(pred)) or "_"
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(ordered_ids)} done  (A wins: {a_wins})")

    print(f"Final: A (checkpoint-a) won {a_wins}/{len(ordered_ids)} clips")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Target"])
        for id_ in ordered_ids:
            writer.writerow([id_, predictions[id_]])

    print(f"Wrote {len(ordered_ids)} rows to {args.output}")


if __name__ == "__main__":
    main()
