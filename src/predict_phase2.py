"""Generate a Zindi Phase 2 submission.

Phase 2 provides no language (or any other) metadata -- IDs look like
ID_TBDTM with no lin_/sna_/lug_ prefix, and audio is delivered as a local
directory of .wav files rather than through the google/WaxalNLP HF dataset.
This script language-IDs each clip with facebook/mms-lid-126 (restricted to
just our 3 target languages, since the true label is known a priori to be
one of them), then routes it to the matching per-language checkpoint for
transcription via the same model-loading/decoding logic (including KenLM
fusion) as src/predict_submission.py.

Usage:
    python -m src.predict_phase2 \
        --test-csv  zindi_data/Test_phase2.csv \
        --audio-dir zindi_data/phase2_audio/audio \
        --output    zindi_data/submission_phase2.csv \
        --lin-checkpoint /data/hf_cache/jgibson2/waxal/checkpoints/finetune_lingala_mms_1b_lowlr/best_model \
        --sna-checkpoint /data/hf_cache/jgibson2/waxal/checkpoints/finetune_shona_mms_300m/best_model \
        --lug-checkpoint /data/hf_cache/jgibson2/waxal/checkpoints/finetune_luganda_mms_1b/best_model
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import soundfile as sf
import torch
from datasets import Audio, Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from src.predict_submission import _load_model, _predict

_LID_MODEL = "facebook/mms-lid-126"
_LANGS = ("lin", "sna", "lug")


def identify_languages(ids: list[str], audio_dir: Path, device: torch.device) -> dict[str, str]:
    """Return {id: lang_code} for lang_code in _LANGS, using MMS-LID restricted
    to just our 3 candidates (the true label is known a priori to be one of
    them, so there's no reason to risk misrouting to an irrelevant 4th
    language among the model's other 123 labels)."""
    processor = AutoFeatureExtractor.from_pretrained(_LID_MODEL)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(_LID_MODEL).to(device)
    model.eval()
    label_indices = {lab: idx for idx, lab in model.config.id2label.items() if lab in _LANGS}
    assert set(label_indices) == set(_LANGS), f"MMS-LID missing labels: {set(_LANGS) - set(label_indices)}"

    id_to_lang = {}
    with torch.inference_mode():
        for id_ in ids:
            array, sampling_rate = sf.read(str(audio_dir / f"{id_}.wav"))
            inputs = processor(array, sampling_rate=sampling_rate, return_tensors="pt")
            logits = model(input_values=inputs.input_values.to(device)).logits[0]
            restricted = {lab: logits[idx].item() for lab, idx in label_indices.items()}
            id_to_lang[id_] = max(restricted, key=restricted.get)
    return id_to_lang


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="zindi_data/Test_phase2.csv")
    parser.add_argument("--audio-dir", default="zindi_data/phase2_audio/audio")
    parser.add_argument("--output", default="zindi_data/submission_phase2.csv")
    parser.add_argument("--lin-checkpoint", required=True)
    parser.add_argument("--sna-checkpoint", required=True)
    parser.add_argument("--lug-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_dir = Path(args.audio_dir)
    checkpoints = {"lin": args.lin_checkpoint, "sna": args.sna_checkpoint, "lug": args.lug_checkpoint}

    with open(args.test_csv) as f:
        ordered_ids = [r["ID"] for r in csv.DictReader(f)]

    print(f"Identifying language for {len(ordered_ids)} clips with {_LID_MODEL}...")
    id_to_lang = identify_languages(ordered_ids, audio_dir, device)
    from collections import Counter
    print("Language distribution:", dict(Counter(id_to_lang.values())))

    id_to_pred: dict[str, str] = {}
    for lang in _LANGS:
        lang_ids = [i for i in ordered_ids if id_to_lang[i] == lang]
        if not lang_ids:
            continue
        print(f"[{lang}] loading checkpoint {checkpoints[lang]}")
        model, processor = _load_model(checkpoints[lang], device)
        ds = Dataset.from_dict({
            "id": lang_ids,
            "audio": [str(audio_dir / f"{i}.wav") for i in lang_ids],
            "transcription": [""] * len(lang_ids),
        }).cast_column("audio", Audio(sampling_rate=16000))
        print(f"[{lang}] running inference on {len(ds)} examples")
        id_to_pred.update(_predict(model, processor, ds, args.batch_size, device, lang))
        del model

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Target"])
        for id_ in ordered_ids:
            writer.writerow([id_, id_to_pred[id_]])

    print(f"Wrote {len(ordered_ids)} rows to {args.output}")


if __name__ == "__main__":
    main()
