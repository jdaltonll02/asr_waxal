"""Generate a Zindi Phase 2 submission from three per-language MMS-300M checkpoints.

Each clip is routed to its language checkpoint based on the language tag in the
Test_phase2.csv ID prefix (lin_, sna_, lug_).  Falls back to the multilingual
checkpoint for any unrecognised prefix.

Usage:
    python -m src.predict_phase2_per_lang \\
        --test-csv  zindi_data/Test_phase2.csv \\
        --audio-dir zindi_data/phase2_audio/audio \\
        --lingala-ckpt  /data/hf_cache/jgibson2/waxal/checkpoints/finetune_lingala_mms_300m_v2/best_model \\
        --shona-ckpt    /data/hf_cache/jgibson2/waxal/checkpoints/finetune_shona_mms_300m_v2/best_model \\
        --luganda-ckpt  /data/hf_cache/jgibson2/waxal/checkpoints/finetune_luganda_mms_300m_v2/best_model \\
        --output        zindi_data/submission_phase2_per_lang_v2.csv
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

try:
    from pyctcdecode import build_ctcdecoder
    _PYCTCDECODE = True
except ImportError:
    _PYCTCDECODE = False

_BATCH_SIZE = 8

# Map ID prefix → checkpoint key
_PREFIX_MAP = {
    "lin": "lingala",
    "sna": "shona",
    "lug": "luganda",
}


def _load_audio(path: Path) -> np.ndarray:
    array, sr = sf.read(str(path))
    if sr != 16000:
        import resampy
        array = resampy.resample(array, sr, 16000)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array.astype(np.float32)


def _build_decoder(processor: Wav2Vec2Processor):
    if not _PYCTCDECODE:
        return None
    vocab = processor.tokenizer.get_vocab()
    labels = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
    labels[processor.tokenizer.pad_token_id] = ""
    seen: set[str] = set()
    for i, lbl in enumerate(labels):
        if lbl in seen:
            labels[i] = f"<dup{i}>"
        seen.add(lbl)
    return build_ctcdecoder(labels)


def _load_model(ckpt: str, device: torch.device):
    processor = Wav2Vec2Processor.from_pretrained(ckpt)
    model = Wav2Vec2ForCTC.from_pretrained(ckpt).to(device).eval()
    decoder = _build_decoder(processor)
    return processor, model, decoder


def _decode_batch(processor, model, decoder, arrays, device):
    inputs = processor(arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(
            inputs.input_values.to(device),
            attention_mask=inputs.attention_mask.to(device),
        ).logits
    if decoder is not None:
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
        return decoder.decode_batch(None, log_probs, beam_width=100)
    return processor.batch_decode(torch.argmax(logits, dim=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="zindi_data/Test_phase2.csv")
    parser.add_argument("--audio-dir", default="zindi_data/phase2_audio/audio")
    parser.add_argument("--lingala-ckpt", required=True)
    parser.add_argument("--shona-ckpt", required=True)
    parser.add_argument("--luganda-ckpt", required=True)
    parser.add_argument("--output", default="zindi_data/submission_phase2_per_lang_v2.csv")
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_dir = Path(args.audio_dir)

    with open(args.test_csv) as f:
        ordered_ids = [r["ID"] for r in csv.DictReader(f)]

    ckpt_map = {
        "lingala": args.lingala_ckpt,
        "shona": args.shona_ckpt,
        "luganda": args.luganda_ckpt,
    }

    # Group IDs by language
    lang_ids: dict[str, list[str]] = {k: [] for k in ckpt_map}
    for id_ in ordered_ids:
        prefix = id_.split("_")[0].lower()
        lang = _PREFIX_MAP.get(prefix)
        if lang is None:
            # Fallback: assign to lingala (shouldn't happen in practice)
            print(f"WARNING: unrecognised prefix '{prefix}' for ID {id_}, defaulting to lingala")
            lang = "lingala"
        lang_ids[lang].append(id_)

    predictions: dict[str, str] = {}

    for lang, ids in lang_ids.items():
        if not ids:
            continue
        print(f"\n--- {lang}: {len(ids)} clips ---")
        processor, model, decoder = _load_model(ckpt_map[lang], device)

        batch_ids, batch_arrays = [], []

        def flush():
            if not batch_ids:
                return
            preds = _decode_batch(processor, model, decoder, batch_arrays, device)
            for id_, pred in zip(batch_ids, preds):
                predictions[id_] = _derepeat(normalize_text(pred)) or "_"
            batch_ids.clear()
            batch_arrays.clear()

        for n, id_ in enumerate(ids):
            batch_ids.append(id_)
            batch_arrays.append(_load_audio(audio_dir / f"{id_}.wav"))
            if len(batch_ids) >= args.batch_size:
                flush()
            if (n + 1) % 200 == 0:
                print(f"  {n + 1}/{len(ids)} done")
        flush()

        # Free GPU memory before loading next model
        del model, processor, decoder
        torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Target"])
        for id_ in ordered_ids:
            writer.writerow([id_, predictions[id_]])

    print(f"\nWrote {len(ordered_ids)} rows to {args.output}")


if __name__ == "__main__":
    main()
