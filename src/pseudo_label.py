"""Generate pseudo-labels for the WAXAL unlabeled split.

Runs the multilingual MMS-300M checkpoint on all three language unlabeled
splits, filters by mean max-softmax confidence, and saves the result as a
HuggingFace dataset to disk for use in self-training.

Usage:
    python -m src.pseudo_label \
        --checkpoint /data/hf_cache/jgibson2/waxal/checkpoints/finetune_multilingual_mms_300m/best_model \
        --output-dir /data/hf_cache/jgibson2/waxal/pseudo_labels/multilingual_mms_300m \
        --confidence-threshold 0.6
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, concatenate_datasets, load_dataset, load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.utils.metrics import normalize_text

_CONFIGS = ["lin_asr", "sna_asr", "lug_asr"]
_CACHE_DIR = "/data/hf_cache/jgibson2/waxal/cache"


def _mean_max_softmax(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    return probs.max(dim=-1).values.mean().item()


def pseudo_label_split(unlabeled, processor, model, device, threshold: float, text_column: str, batch_size: int = 4):
    accepted_indices, transcriptions, confidences = [], [], []
    batch_indices, batch_arrays = [], []

    _MAX_DURATION_SEC = 60.0

    def _run_batch(indices, arrays):
        resampled = [arr.astype(np.float32) for arr in arrays]
        inputs = processor(resampled, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device),
                           attention_mask=inputs.attention_mask.to(device)).logits
        for idx, lgts in zip(indices, logits):
            conf = _mean_max_softmax(lgts.unsqueeze(0))
            if conf >= threshold:
                pred = processor.batch_decode(torch.argmax(lgts.unsqueeze(0), dim=-1))[0]
                pred = normalize_text(pred).strip()
                if pred:
                    accepted_indices.append(idx)
                    transcriptions.append(pred)
                    confidences.append(conf)

    def flush():
        if not batch_indices:
            return
        try:
            _run_batch(batch_indices, batch_arrays)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache() if device.type == "cuda" else None
            print(f"    WARNING: OOM on batch containing indices {batch_indices}; retrying one-by-one")
            for idx, arr in zip(batch_indices, batch_arrays):
                try:
                    _run_batch([idx], [arr])
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache() if device.type == "cuda" else None
                    print(f"    WARNING: skipping index {idx} (OOM even alone, {len(arr) / 16000:.1f}s clip)")
        batch_indices.clear()
        batch_arrays.clear()

    for i in range(len(unlabeled)):
        raw = unlabeled[i]["audio"]
        arr = np.array(raw["array"], dtype=np.float32)
        sr = raw["sampling_rate"]
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        if len(arr) / 16000 > _MAX_DURATION_SEC:
            print(f"    Skipping index {i}: clip too long ({len(arr) / 16000:.1f}s > {_MAX_DURATION_SEC}s)")
            continue
        batch_indices.append(i)
        batch_arrays.append(arr)
        if len(batch_indices) >= batch_size:
            flush()
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(unlabeled)} processed, {len(accepted_indices)} accepted so far")
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
    flush()

    if not accepted_indices:
        return None

    subset = unlabeled.select(accepted_indices)
    subset = subset.remove_columns([text_column])
    subset = subset.add_column(text_column, transcriptions)
    subset = subset.add_column("pseudo_confidence", confidences)
    subset = subset.cast_column("audio", Audio(sampling_rate=16000))
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--configs", nargs="+", default=None,
                         help="Process only these configs (e.g. one language per process, "
                              "to keep peak memory bounded to a single split). Defaults to all.")
    parser.add_argument("--combine-only", action="store_true",
                         help="Skip processing; just concatenate already-saved per-language parts.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = args.configs or _CONFIGS

    if not args.combine_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
        model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint).to(device).eval()

        for cfg in configs:
            part_path = out / f"_part_{cfg}"
            if (part_path / "dataset_info.json").exists():
                print(f"Skipping {cfg} unlabeled split (already saved at {part_path})")
                continue
            print(f"Processing {cfg} unlabeled split...")
            ds = load_dataset("google/WaxalNLP", cfg, split="unlabeled", cache_dir=_CACHE_DIR)
            labeled = pseudo_label_split(ds, processor, model, device, args.confidence_threshold, "transcription", args.batch_size)
            if labeled is not None:
                labeled.save_to_disk(str(part_path))
                print(f"  {cfg}: {len(labeled)}/{len(ds)} accepted (threshold={args.confidence_threshold}), saved to {part_path}")
            else:
                print(f"  {cfg}: 0 accepted")
        return

    part_paths = [out / f"_part_{cfg}" for cfg in _CONFIGS if (out / f"_part_{cfg}" / "dataset_info.json").exists()]
    if not part_paths:
        print("No pseudo-labels generated. Lower --confidence-threshold.")
        return

    parts = [load_from_disk(str(p)).cast_column("audio", Audio(sampling_rate=16000)) for p in part_paths]
    combined = concatenate_datasets(parts)
    combined.save_to_disk(str(out / "combined"))
    print(f"Saved {len(combined)} pseudo-labeled examples to {out / 'combined'}")


if __name__ == "__main__":
    main()
