import torch, soundfile as sf
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import csv

model_id = "facebook/mms-lid-126"
processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to("cuda")
model.eval()
label_indices = {lab: idx for idx, lab in model.config.id2label.items() if lab in ("lin", "sna", "lug")}
print("indices:", label_indices)

with open("zindi_data/Test_phase2.csv") as f:
    all_ids = [r["ID"] for r in csv.DictReader(f)][:10]

for f in all_ids:
    array, sr = sf.read(f"zindi_data/phase2_audio/audio/{f}.wav")
    inputs = processor(array, sampling_rate=sr, return_tensors="pt")
    with torch.inference_mode():
        logits = model(input_values=inputs.input_values.to("cuda")).logits[0]
    probs = torch.softmax(logits, dim=-1)
    restricted = {lab: round(probs[idx].item(), 4) for lab, idx in label_indices.items()}
    top5 = torch.topk(probs, 5)
    top5_labels = [(model.config.id2label[i.item()], round(p.item(),4)) for p, i in zip(top5.values, top5.indices)]
    print(f, "restricted probs:", restricted, " | global top5:", top5_labels)
