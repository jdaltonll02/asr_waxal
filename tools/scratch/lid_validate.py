import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from datasets import Audio, load_dataset

model_id = "facebook/mms-lid-126"
processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to("cuda")
model.eval()
id2label = model.config.id2label
our_langs = {lab: idx for idx, lab in id2label.items() if lab in ("lin", "sna", "lug")}
print("label indices:", our_langs)

correct = {"lin_asr": 0, "sna_asr": 0, "lug_asr": 0}
total = {"lin_asr": 0, "sna_asr": 0, "lug_asr": 0}
expected = {"lin_asr": "lin", "sna_asr": "sna", "lug_asr": "lug"}
for cfg in ["lin_asr", "sna_asr", "lug_asr"]:
    ds = load_dataset("google/WaxalNLP", cfg, split="test[:30]", cache_dir="/data/hf_cache/jgibson2/waxal/cache")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    for ex in ds:
        inputs = processor(ex["audio"]["array"], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(input_values=inputs.input_values.to("cuda")).logits[0]
        restricted = {lab: logits[idx].item() for lab, idx in our_langs.items()}
        pred = max(restricted, key=restricted.get)
        total[cfg] += 1
        if pred == expected[cfg]:
            correct[cfg] += 1
for cfg in correct:
    print(f"{cfg}: {correct[cfg]}/{total[cfg]} correct")
