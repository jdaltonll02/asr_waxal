"""Generate a Zindi Phase 2 submission using LM+acoustic combined-score
language ID: for each clip, decode with each of the 3 per-language checkpoints
via KenLM-fused beam search, and pick whichever produces the highest
length-normalized (logit_score + lm_score) -- i.e. whichever candidate
language's decode is both acoustically confident AND linguistically fluent.

This replaced two earlier approaches that both failed to discriminate on a
large fraction of the real Phase 2 clips:
  - facebook/mms-lid-126 (external LID): validated at 97.8% on Phase 1 audio,
    but its unrestricted top-1 wasn't even one of our 3 languages for 55% of
    Phase 2 clips (mostly confusing WAXAL's noisy natural-environment
    recordings with 'luo').
  - raw ASR confidence (mean max-softmax-probability): validated at 96.7% on
    Phase 1 audio, but produced near-identical (~1% spread) scores across all
    3 languages for roughly half of Phase 2's clips -- an uninformative,
    near-arbitrary tiebreak, not a real signal.
This combined score validated at 98.9% on Phase 1 audio with decisive
(non-tied) margins even on its one miss -- see tools/scratch/lm_lid_validate.py.
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
    parser.add_argument("--output", default="zindi_data/submission_phase2_lmlid.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_dir = Path(args.audio_dir)

    with open(args.test_csv) as f:
        ordered_ids = [r["ID"] for r in csv.DictReader(f)]

    print("Loading checkpoints + LM-fused decoders...")
    models = {lang: _load_model(ckpt, device) for lang, ckpt in CHECKPOINTS.items()}
    decoders = {lang: _build_ctc_decoder(models[lang][1], lang) for lang in CHECKPOINTS}

    id_to_pred: dict[str, str] = {}
    id_to_lang: dict[str, str] = {}
    for n, id_ in enumerate(ordered_ids):
        array, sr = sf.read(str(audio_dir / f"{id_}.wav"))
        scores, texts = {}, {}
        for lang, (model, processor) in models.items():
            inputs = processor(array, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.inference_mode():
                logits = model(input_values=inputs.input_values.to(device),
                               attention_mask=inputs.attention_mask.to(device)).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0].cpu().numpy()
            decoder, beam_width = decoders[lang]
            beams = decoder.decode_beams(log_probs, beam_width=beam_width)
            text, _, _, logit_score, lm_score = beams[0]
            length = max(len(text.split()), 1)
            scores[lang] = (logit_score + lm_score) / length
            texts[lang] = text

        winner = max(scores, key=scores.get)
        id_to_lang[id_] = winner
        id_to_pred[id_] = _derepeat(normalize_text(texts[winner])) or "_"

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
