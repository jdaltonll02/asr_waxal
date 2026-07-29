"""Generate a Zindi Phase 2 submission using ASR-confidence-based language ID
instead of an external LID model.

facebook/mms-lid-126 validated at 97.8% on known-language Phase 1 audio, but
on the real Phase 2 clips its own top-1 (unrestricted) prediction was often a
totally unrelated language (e.g. 'luo', even 'tha' with near-zero confidence
for any of our 3 candidates) -- a domain-generalization failure specific to
Phase 2's recording conditions, not a Phase-1-representative failure mode.
This instead runs all 3 fine-tuned per-language checkpoints on every clip and
picks whichever is most confident (mean max-softmax-probability per frame),
since those checkpoints are validated deeply on this exact corpus/domain.
The winning model's logits are decoded immediately (with KenLM fusion) rather
than re-running inference, since the confidence pass already produced them.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import soundfile as sf
import torch

from src.predict_submission import _build_ctc_decoder, _derepeat, _load_model
from src.utils.metrics import normalize_text

CHECKPOINTS = {
    "lin": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_lingala_mms_1b_lowlr/best_model",
    "sna": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_shona_mms_300m/best_model",
    "lug": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_luganda_mms_1b/best_model",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="zindi_data/Test_phase2.csv")
    parser.add_argument("--audio-dir", default="zindi_data/phase2_audio/audio")
    parser.add_argument("--output", default="zindi_data/submission_phase2_asrlid.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_dir = Path(args.audio_dir)

    with open(args.test_csv) as f:
        ordered_ids = [r["ID"] for r in csv.DictReader(f)]

    print("Loading all 3 checkpoints...")
    models = {lang: _load_model(ckpt, device) for lang, ckpt in CHECKPOINTS.items()}
    decoders = {lang: _build_ctc_decoder(models[lang][1], lang) for lang in CHECKPOINTS}

    id_to_pred: dict[str, str] = {}
    id_to_lang: dict[str, str] = {}
    for n, id_ in enumerate(ordered_ids):
        array, sr = sf.read(str(audio_dir / f"{id_}.wav"))
        scores = {}
        logits_per_lang = {}
        for lang, (model, processor) in models.items():
            inputs = processor(array, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.inference_mode():
                logits = model(input_values=inputs.input_values.to(device),
                               attention_mask=inputs.attention_mask.to(device)).logits
            probs = torch.softmax(logits, dim=-1)
            scores[lang] = probs.max(dim=-1).values.mean().item()
            logits_per_lang[lang] = logits
        winner = max(scores, key=scores.get)
        id_to_lang[id_] = winner

        decoder, beam_width = decoders[winner]
        log_probs = torch.nn.functional.log_softmax(logits_per_lang[winner], dim=-1).cpu().numpy()
        pred = decoder.decode_batch(None, log_probs, beam_width=beam_width)[0]
        id_to_pred[id_] = _derepeat(normalize_text(pred)) or "_"

        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(ordered_ids)} done")

    from collections import Counter
    print("Language distribution:", dict(Counter(id_to_lang.values())))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Target"])
        for id_ in ordered_ids:
            writer.writerow([id_, id_to_pred[id_]])
    print(f"Wrote {len(ordered_ids)} rows to {args.output}")


if __name__ == "__main__":
    main()
