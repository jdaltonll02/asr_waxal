"""Generate a Zindi Phase 2 submission from three per-language MMS-300M checkpoints.

Each clip is routed to its language checkpoint using ASR-confidence LID:
all three models run on every clip and the one with the highest mean
max-softmax probability wins.  This avoids relying on ID prefixes (which
are all 'ID_' in Phase 2) or an external LID model.

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


def _load_audio(path: Path) -> np.ndarray:
    array, sr = sf.read(str(path))
    if sr != 16000:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
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


def _asr_lid_score(model, processor, array: np.ndarray, device: torch.device) -> float:
    """Mean max-softmax probability — higher = model is more confident on this clip."""
    inputs = processor([array], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(
            inputs.input_values.to(device),
            attention_mask=inputs.attention_mask.to(device),
        ).logits
    return torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item(), logits


def _decode_logits(logits, processor, decoder):
    if decoder is not None:
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
        return decoder.decode_batch(None, log_probs, beam_width=100)[0]
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="zindi_data/Test_phase2.csv")
    parser.add_argument("--audio-dir", default="zindi_data/phase2_audio/audio")
    parser.add_argument("--lingala-ckpt", required=True)
    parser.add_argument("--shona-ckpt", required=True)
    parser.add_argument("--luganda-ckpt", required=True)
    parser.add_argument("--output", default="zindi_data/submission_phase2_per_lang_v2.csv")
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

    print("Loading all 3 checkpoints...")
    models = {lang: _load_model(ckpt, device) for lang, ckpt in ckpt_map.items()}

    predictions: dict[str, str] = {}
    lang_counts: dict[str, int] = {k: 0 for k in ckpt_map}

    for n, id_ in enumerate(ordered_ids):
        array = _load_audio(audio_dir / f"{id_}.wav")

        # ASR-confidence LID: pick the model with highest mean max-softmax prob
        best_lang, best_score, best_logits = None, -1.0, None
        for lang, (processor, model, _) in models.items():
            score, logits = _asr_lid_score(model, processor, array, device)
            if score > best_score:
                best_lang, best_score, best_logits = lang, score, logits

        processor, _, decoder = models[best_lang]
        pred = _decode_logits(best_logits, processor, decoder)
        predictions[id_] = _derepeat(normalize_text(pred)) or "_"
        lang_counts[best_lang] += 1

        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(ordered_ids)} done  lang_counts={lang_counts}")

    print(f"Language distribution: {lang_counts}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Target"])
        for id_ in ordered_ids:
            writer.writerow([id_, predictions[id_]])

    print(f"Wrote {len(ordered_ids)} rows to {args.output}")


if __name__ == "__main__":
    main()
