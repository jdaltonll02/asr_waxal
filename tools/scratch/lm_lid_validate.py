"""Validate LM+acoustic combined-score language ID: for each candidate
checkpoint, run CTC forward pass, then LM-fused beam search decode, and score
by the top beam's (logit_score + lm_score) -- this captures both acoustic
confidence AND whether the decoded text is fluent in that language, unlike
raw mean-max-softmax-probability which was found to barely discriminate on
ambiguous Phase 2 clips (near-identical ~1% spread across all 3 languages).
Validate against known-language Phase 1 audio before trusting it on Phase 2.
"""
import torch
from datasets import Audio, load_dataset

from src.predict_submission import _load_model, _build_ctc_decoder

CHECKPOINTS = {
    "lin": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_lingala_mms_1b_lowlr/best_model",
    "sna": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_shona_mms_300m/best_model",
    "lug": "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_luganda_mms_1b/best_model",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading checkpoints + LM-fused decoders...")
models = {lang: _load_model(ckpt, device) for lang, ckpt in CHECKPOINTS.items()}
decoders = {lang: _build_ctc_decoder(models[lang][1], lang) for lang in CHECKPOINTS}


def score(array, lang):
    model, processor = models[lang]
    inputs = processor(array, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(input_values=inputs.input_values.to(device),
                        attention_mask=inputs.attention_mask.to(device)).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0].cpu().numpy()
    decoder, beam_width = decoders[lang]
    beams = decoder.decode_beams(log_probs, beam_width=beam_width)
    text, _, _, logit_score, lm_score = beams[0]
    # normalize by decoded length so longer/shorter outputs aren't unfairly
    # penalized/rewarded relative to each other
    length = max(len(text.split()), 1)
    return (logit_score + lm_score) / length, text


correct = {"lin_asr": 0, "sna_asr": 0, "lug_asr": 0}
total = {"lin_asr": 0, "sna_asr": 0, "lug_asr": 0}
expected = {"lin_asr": "lin", "sna_asr": "sna", "lug_asr": "lug"}
for cfg in ["lin_asr", "sna_asr", "lug_asr"]:
    ds = load_dataset("google/WaxalNLP", cfg, split="test[:30]", cache_dir="/data/hf_cache/jgibson2/waxal/cache")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    for ex in ds:
        array = ex["audio"]["array"]
        scores = {}
        for lang in CHECKPOINTS:
            s, _ = score(array, lang)
            scores[lang] = s
        pred = max(scores, key=scores.get)
        total[cfg] += 1
        if pred == expected[cfg]:
            correct[cfg] += 1
        else:
            print(f"  MISS {cfg} predicted {pred}: scores={ {k: round(v,3) for k,v in scores.items()} }")
for cfg in correct:
    print(f"{cfg}: {correct[cfg]}/{total[cfg]} correct")
