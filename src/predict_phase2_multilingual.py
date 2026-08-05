"""Generate a Zindi Phase 2 submission using a single multilingual checkpoint.

No language ID is needed — the multilingual model handles all three languages
from the audio signal alone. Audio is read from a local directory of .wav files.

Usage:
    python -m src.predict_phase2_multilingual \
        --test-csv  zindi_data/Test_phase2.csv \
        --audio-dir zindi_data/phase2_audio/audio \
        --checkpoint /data/hf_cache/jgibson2/waxal/checkpoints/finetune_multilingual_mms_300m/best_model \
        --output    zindi_data/submission_phase2_multilingual.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperForConditionalGeneration, WhisperProcessor

from src.predict_submission import _derepeat, _is_ctc
from src.utils.metrics import normalize_text

try:
    from pyctcdecode import build_ctcdecoder
    _PYCTCDECODE = True
except ImportError:
    _PYCTCDECODE = False

_BATCH_SIZE = 8
_LM_PATH = Path("/data/hf_cache/jgibson2/waxal/lm/multilingual.binary")
_LM_CORPUS = Path("/data/hf_cache/jgibson2/waxal/lm/multilingual.txt")


def _load_audio(path: Path) -> np.ndarray:
    array, sr = sf.read(str(path))
    if sr != 16000:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="zindi_data/Test_phase2.csv")
    parser.add_argument("--audio-dir", default="zindi_data/phase2_audio/audio")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="zindi_data/submission_phase2_multilingual.csv")
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE)
    parser.add_argument("--lm-alpha", type=float, default=None,
                        help="pyctcdecode LM weight (alpha). Overrides default 0.0.")
    parser.add_argument("--lm-beta", type=float, default=None,
                        help="pyctcdecode word insertion bonus (beta). Overrides default 0.0.")
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--lm-path", default=None, help="Override the default KenLM binary path.")
    parser.add_argument("--lm-corpus", default=None, help="Override the default KenLM unigram corpus path.")
    args = parser.parse_args()

    global _LM_PATH, _LM_CORPUS
    if args.lm_path:
        _LM_PATH = Path(args.lm_path)
    if args.lm_corpus:
        _LM_CORPUS = Path(args.lm_corpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_dir = Path(args.audio_dir)

    with open(args.test_csv) as f:
        ordered_ids = [r["ID"] for r in csv.DictReader(f)]

    is_ctc = _is_ctc(args.checkpoint)
    if is_ctc:
        processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
        model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint).to(device).eval()
        if _PYCTCDECODE:
            vocab = processor.tokenizer.get_vocab()
            labels = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
            labels[processor.tokenizer.pad_token_id] = ""
            # pyctcdecode requires unique labels; deduplicate by blanking repeats
            seen = set()
            for i, lbl in enumerate(labels):
                if lbl in seen:
                    labels[i] = f"<dup{i}>"
                seen.add(lbl)
            kw = {}
            if args.lm_alpha is not None:
                kw["alpha"] = args.lm_alpha
            if args.lm_beta is not None:
                kw["beta"] = args.lm_beta
            if _LM_PATH.exists():
                unigrams = {word for line in open(_LM_CORPUS, encoding="utf-8") for word in line.split()}
                decoder = build_ctcdecoder(labels, kenlm_model_path=str(_LM_PATH), unigrams=unigrams, **kw)
                print(f"Using KenLM fusion: {_LM_PATH}  alpha={kw.get('alpha', 0.5)}  beta={kw.get('beta', 1.5)}")
            else:
                decoder = build_ctcdecoder(labels, **kw)
                print(f"No KenLM found at {_LM_PATH}; pyctcdecode alpha={kw.get('alpha', 0.0)}  beta={kw.get('beta', 0.0)}")
        else:
            decoder = None
    else:
        processor = WhisperProcessor.from_pretrained(args.checkpoint)
        model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint).to(device).eval()
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []

    predictions: dict[str, str] = {}
    batch_ids, batch_arrays = [], []

    def flush():
        if not batch_ids:
            return
        if is_ctc:
            inputs = processor(batch_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.inference_mode():
                logits = model(inputs.input_values.to(device),
                               attention_mask=inputs.attention_mask.to(device)).logits
            if decoder is not None:
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
                preds = decoder.decode_batch(None, log_probs, beam_width=args.beam_width)
            else:
                preds = processor.batch_decode(torch.argmax(logits, dim=-1))
        else:
            inputs = processor.feature_extractor(batch_arrays, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                generated = model.generate(inputs.input_features.to(device),
                                           do_sample=False, num_beams=1, no_repeat_ngram_size=4)
            preds = processor.batch_decode(generated, skip_special_tokens=True)
        for id_, pred in zip(batch_ids, preds):
            predictions[id_] = _derepeat(normalize_text(pred)) or "_"
        batch_ids.clear()
        batch_arrays.clear()

    for n, id_ in enumerate(ordered_ids):
        batch_ids.append(id_)
        batch_arrays.append(_load_audio(audio_dir / f"{id_}.wav"))
        if len(batch_ids) >= args.batch_size:
            flush()
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(ordered_ids)} done")
    flush()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Target"])
        for id_ in ordered_ids:
            writer.writerow([id_, predictions[id_]])

    print(f"Wrote {len(ordered_ids)} rows to {args.output}")


if __name__ == "__main__":
    main()
