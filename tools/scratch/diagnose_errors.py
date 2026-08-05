"""Quick diagnostic: what kind of WER errors dominate for the largebatch checkpoint
on WAXAL validation data (labeled) -- insertions, deletions, substitutions, or
word-boundary/splitting issues? Informs which lever is worth pursuing further.
"""
import numpy as np
import soundfile as sf
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import jiwer

from src.utils.metrics import normalize_text

try:
    from pyctcdecode import build_ctcdecoder
except ImportError:
    raise SystemExit("pyctcdecode required")

CKPT = "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_multilingual_mms_300m_largebatch/best_model"
LM_PATH = "/data/hf_cache/jgibson2/waxal/lm/multilingual.binary"
LM_CORPUS = "/data/hf_cache/jgibson2/waxal/lm/multilingual.txt"
CACHE = "/data/hf_cache/jgibson2/waxal/cache"
N_SAMPLES = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained(CKPT)
model = Wav2Vec2ForCTC.from_pretrained(CKPT).to(device).eval()

vocab = processor.tokenizer.get_vocab()
labels = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
labels[processor.tokenizer.pad_token_id] = ""
seen = set()
for i, lbl in enumerate(labels):
    if lbl in seen:
        labels[i] = f"<dup{i}>"
    seen.add(lbl)
unigrams = {word for line in open(LM_CORPUS, encoding="utf-8") for word in line.split()}
decoder = build_ctcdecoder(labels, kenlm_model_path=LM_PATH, unigrams=unigrams, alpha=0.0, beta=0.5)

splits = [load_dataset("google/WaxalNLP", cfg, split="validation", cache_dir=CACHE) for cfg in ["lin_asr", "sna_asr", "lug_asr"]]
val_ds = concatenate_datasets(splits).select(range(N_SAMPLES))

refs, hyps = [], []
for ex in val_ds:
    raw = ex["audio"]
    arr = np.array(raw["array"], dtype=np.float32)
    sr = raw["sampling_rate"]
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    inputs = processor([arr], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
    hyp = decoder.decode_batch(None, log_probs, beam_width=100)[0]
    refs.append(normalize_text(ex["transcription"]))
    hyps.append(normalize_text(hyp))

out = jiwer.process_words(refs, hyps)
print(f"WER: {out.wer:.4f}")
print(f"Substitutions: {out.substitutions}  Deletions: {out.deletions}  Insertions: {out.insertions}  Hits: {out.hits}")

# Check specifically for word-boundary issues: cases where ref word count differs a lot from hyp
ref_words = sum(len(r.split()) for r in refs)
hyp_words = sum(len(h.split()) for h in hyps)
print(f"Total ref words: {ref_words}  Total hyp words: {hyp_words}  ratio: {hyp_words/ref_words:.3f}")

print("\nSample alignments (first 8):")
for i in range(min(8, len(refs))):
    print(f"REF: {refs[i]}")
    print(f"HYP: {hyps[i]}")
    print()
