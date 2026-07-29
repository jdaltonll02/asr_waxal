"""Compare external MMS-LID vs ASR-confidence routing decisions per-clip on
the real Phase 2 test set, to diagnose how much routing uncertainty exists."""
import csv
import torch
import soundfile as sf
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from src.predict_submission import _load_model

CHECKPOINTS = {
    "lin": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_lingala_mms_1b_lowlr/best_model",
    "sna": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_shona_mms_300m/best_model",
    "lug": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_luganda_mms_1b/best_model",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading MMS-LID...")
lid_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-126")
lid_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-126").to(device)
lid_model.eval()
lid_label_indices = {lab: idx for idx, lab in lid_model.config.id2label.items() if lab in CHECKPOINTS}

print("Loading ASR checkpoints...")
asr_models = {lang: _load_model(ckpt, device) for lang, ckpt in CHECKPOINTS.items()}

with open("zindi_data/Test_phase2.csv") as f:
    ids = [r["ID"] for r in csv.DictReader(f)]

rows = []
for n, id_ in enumerate(ids):
    array, sr = sf.read(f"zindi_data/phase2_audio/audio/{id_}.wav")

    lid_inputs = lid_processor(array, sampling_rate=sr, return_tensors="pt")
    with torch.inference_mode():
        lid_logits = lid_model(input_values=lid_inputs.input_values.to(device)).logits[0]
    lid_probs = torch.softmax(lid_logits, dim=-1)
    lid_restricted = {lab: lid_probs[idx].item() for lab, idx in lid_label_indices.items()}
    lid_pred = max(lid_restricted, key=lid_restricted.get)
    lid_conf = lid_restricted[lid_pred]
    lid_unrestricted_top1 = lid_model.config.id2label[lid_probs.argmax().item()]

    asr_scores = {}
    for lang, (model, processor) in asr_models.items():
        inputs = processor(array, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = model(input_values=inputs.input_values.to(device),
                            attention_mask=inputs.attention_mask.to(device)).logits
        probs = torch.softmax(logits, dim=-1)
        asr_scores[lang] = probs.max(dim=-1).values.mean().item()
    asr_pred = max(asr_scores, key=asr_scores.get)

    rows.append({
        "id": id_, "lid_pred": lid_pred, "lid_conf": round(lid_conf, 4),
        "lid_unrestricted_top1": lid_unrestricted_top1,
        "asr_pred": asr_pred, "asr_scores": {k: round(v, 4) for k, v in asr_scores.items()},
        "agree": lid_pred == asr_pred,
    })
    if (n + 1) % 200 == 0:
        print(f"  {n + 1}/{len(ids)} done")

agree = sum(r["agree"] for r in rows)
print(f"\nAgreement: {agree}/{len(rows)} ({100*agree/len(rows):.1f}%)")

from collections import Counter
print("LID distribution:", dict(Counter(r["lid_pred"] for r in rows)))
print("ASR distribution:", dict(Counter(r["asr_pred"] for r in rows)))

# how often does LID's true top-1 (unrestricted) fall outside our 3 languages entirely?
off_target = sum(1 for r in rows if r["lid_unrestricted_top1"] not in CHECKPOINTS)
print(f"LID's unrestricted top-1 is NOT one of our 3 languages: {off_target}/{len(rows)} ({100*off_target/len(rows):.1f}%)")

with open("tools/scratch/lid_agreement.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "lid_pred", "lid_conf", "lid_unrestricted_top1", "asr_pred", "asr_scores", "agree"])
    writer.writeheader()
    writer.writerows(rows)
print("Wrote tools/scratch/lid_agreement.csv")
