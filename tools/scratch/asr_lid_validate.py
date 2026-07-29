"""Validate ASR-confidence-based language ID: for each candidate checkpoint,
run CTC forward pass and score by mean max-softmax-probability per frame.
Pick whichever of our 3 checkpoints is most confident. Validate against
known-language Phase 1 WAXAL audio before trusting it on Phase 2.
"""
import torch
from datasets import Audio, load_dataset

from src.predict_submission import _load_model

CHECKPOINTS = {
    "lin": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_lingala_mms_1b_lowlr/best_model",
    "sna": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_shona_mms_300m/best_model",
    "lug": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_luganda_mms_1b/best_model",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {lang: _load_model(ckpt, device) for lang, ckpt in CHECKPOINTS.items()}


def confidence(array, lang):
    model, processor = models[lang]
    inputs = processor(array, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(input_values=inputs.input_values.to(device),
                        attention_mask=inputs.attention_mask.to(device)).logits
    probs = torch.softmax(logits, dim=-1)
    return probs.max(dim=-1).values.mean().item()


correct = {"lin_asr": 0, "sna_asr": 0, "lug_asr": 0}
total = {"lin_asr": 0, "sna_asr": 0, "lug_asr": 0}
expected = {"lin_asr": "lin", "sna_asr": "sna", "lug_asr": "lug"}
for cfg in ["lin_asr", "sna_asr", "lug_asr"]:
    ds = load_dataset("google/WaxalNLP", cfg, split="test[:30]", cache_dir="/data/hf_cache/jgibson2/waxal/cache")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    for ex in ds:
        array = ex["audio"]["array"]
        scores = {lang: confidence(array, lang) for lang in CHECKPOINTS}
        pred = max(scores, key=scores.get)
        total[cfg] += 1
        if pred == expected[cfg]:
            correct[cfg] += 1
        else:
            print(f"  MISS {cfg} predicted {pred}: scores={ {k: round(v,4) for k,v in scores.items()} }")
for cfg in correct:
    print(f"{cfg}: {correct[cfg]}/{total[cfg]} correct")
