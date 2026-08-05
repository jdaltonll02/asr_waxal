"""Sweep pyctcdecode alpha/beta on the WAXAL multilingual validation set.

Loads the multilingual MMS-300M checkpoint once, runs CTC logits over the
validation split, then scores every (alpha, beta) grid point by CER.
Writes a CSV of results and prints the best combination.

Usage:
    python -m src.sweep_lm_weights \\
        --checkpoint /data/hf_cache/jgibson2/waxal/checkpoints/finetune_multilingual_mms_300m/best_model \\
        --output     /data/hf_cache/jgibson2/waxal/outputs/lm_sweep/results.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.utils.metrics import normalize_text

try:
    from pyctcdecode import build_ctcdecoder
except ImportError:
    raise SystemExit("pyctcdecode is required: pip install pyctcdecode")

# Grid to sweep
_ALPHAS = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
_BETAS  = [0.0, 0.5, 1.0, 1.5, 2.0]

_BATCH = 8

_CONFIGS = ["lin_asr", "sna_asr", "lug_asr"]
_CACHE = "/data/hf_cache/jgibson2/waxal/cache"
_LM_PATH = Path("/data/hf_cache/jgibson2/waxal/lm/multilingual.binary")
_LM_CORPUS = Path("/data/hf_cache/jgibson2/waxal/lm/multilingual.txt")


def _cer(ref: str, hyp: str) -> float:
    """Character error rate (edit distance / len(ref)), clamped to [0, 1]."""
    import jiwer
    if not ref:
        return 0.0 if not hyp else 1.0
    return min(jiwer.cer(ref, hyp), 1.0)


def _build_decoder(processor: Wav2Vec2Processor, alpha: float, beta: float, unigrams: set[str] | None):
    vocab = processor.tokenizer.get_vocab()
    labels = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
    labels[processor.tokenizer.pad_token_id] = ""
    seen: set[str] = set()
    for i, lbl in enumerate(labels):
        if lbl in seen:
            labels[i] = f"<dup{i}>"
        seen.add(lbl)
    if unigrams is not None:
        return build_ctcdecoder(labels, kenlm_model_path=str(_LM_PATH), unigrams=unigrams, alpha=alpha, beta=beta)
    return build_ctcdecoder(labels, alpha=alpha, beta=beta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="/data/hf_cache/jgibson2/waxal/outputs/lm_sweep/results.csv")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Cap validation samples per language to keep sweep fast")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
    model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint).to(device).eval()

    # Load only the validation split for each language — no cast_column so audio
    # is decoded on demand below via soundfile, avoiding a slow eager decode pass.
    splits = [
        load_dataset("google/WaxalNLP", cfg, split="validation", cache_dir=_CACHE)
        for cfg in _CONFIGS
    ]
    val_ds = concatenate_datasets(splits)
    if args.max_samples and len(val_ds) > args.max_samples:
        val_ds = val_ds.select(range(args.max_samples))
    print(f"Validation samples: {len(val_ds)}")

    # Collect logits once (expensive)
    print("Computing logits...")
    all_log_probs: list[np.ndarray] = []
    all_refs: list[str] = []

    for i in range(0, len(val_ds), _BATCH):
        batch = val_ds.select(range(i, min(i + _BATCH, len(val_ds))))
        arrays = []
        for ex in batch:
            raw = ex["audio"]
            # raw is either already decoded (dict with 'array') or a path string
            if isinstance(raw, dict):
                arr = np.array(raw["array"], dtype=np.float32)
                sr = raw.get("sampling_rate", 16000)
            else:
                arr, sr = sf.read(str(raw))
                arr = arr.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            arrays.append(arr if len(arr) > 0 else np.zeros(160, dtype=np.float32))
        refs = [normalize_text(ex["transcription"]) for ex in batch]
        inputs = processor(arrays, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = model(
                inputs.input_values.to(device),
                attention_mask=inputs.attention_mask.to(device),
            ).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
        # Store per-utterance (unpadded slice not needed for CER sweep — full frame is fine)
        for j in range(len(arrays)):
            all_log_probs.append(log_probs[j])
        all_refs.extend(refs)
        if (i // _BATCH + 1) % 20 == 0:
            print(f"  {i + _BATCH}/{len(val_ds)}")

    print(f"Logits collected for {len(all_log_probs)} utterances. Sweeping grid...")

    unigrams = None
    if _LM_PATH.exists():
        unigrams = {word for line in open(_LM_CORPUS, encoding="utf-8") for word in line.split()}
        print(f"Using KenLM fusion: {_LM_PATH}")
    else:
        print(f"WARNING: no KenLM found at {_LM_PATH}; sweeping with no LM (alpha/beta will have no effect)")

    results = []
    for alpha, beta in itertools.product(_ALPHAS, _BETAS):
        decoder = _build_decoder(processor, alpha, beta, unigrams)
        hyps = decoder.decode_batch(None, all_log_probs, beam_width=100)
        hyps_norm = [normalize_text(h) for h in hyps]
        avg_cer = float(np.mean([_cer(r, h) for r, h in zip(all_refs, hyps_norm)]))
        results.append({"alpha": alpha, "beta": beta, "cer": avg_cer})
        print(f"  alpha={alpha:.1f}  beta={beta:.1f}  CER={avg_cer:.4f}")

    results.sort(key=lambda x: x["cer"])
    best = results[0]
    print(f"\nBest: alpha={best['alpha']}  beta={best['beta']}  CER={best['cer']:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "cer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
