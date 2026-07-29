"""Evaluate our submitted checkpoints against their OWN speaker-disjoint
validation split (reconstructed with the exact same seed/fraction used
during training) instead of WAXAL's official test split, which shares
66-89% of speakers with train. This tests whether there's a real
speaker-generalization gap independent of Phase 2's specific audio domain.
"""
import torch
from torch.utils.data import DataLoader

from src.data.clean_filter import clean_dataset
from src.data.dataset import CTCCollator
from src.data.load_waxal import load_waxal
from src.data.speaker_split import apply_speaker_holdout
from src.models.mms_wrapper import load_mms_1b_finetune
from src.utils.config import resolve_experiment
from src.utils.metrics import compute_metrics

CONFIGS = {
    "lin": ("configs/experiment/finetune_lingala_mms_1b_lowlr.yaml",
            "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_lingala_mms_1b_lowlr/best_model", "lin"),
    "lug": ("configs/experiment/finetune_luganda_mms_1b.yaml",
            "/data/hf_cache/jgibson2/waxal/checkpoints/finetune_luganda_mms_1b/best_model", "lug"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for lang, (config_path, checkpoint, iso) in CONFIGS.items():
    print(f"=== {lang} ===")
    config = resolve_experiment(config_path)
    raw = load_waxal(config["data"])
    held = apply_speaker_holdout(raw, config["data"])
    cleaned, stats = clean_dataset(held, config["data"])
    val_set = cleaned[config["data"]["val_split"]]
    print(f"  speaker-holdout validation set: {len(val_set)} utterances")

    # rebuild the training-time vocab/processor (deterministic from train text)
    # then load the real fine-tuned weights over it, same pattern as evaluate.py
    model, processor = load_mms_1b_finetune(
        config["model"], iso, cleaned[config["data"]["train_split"]],
        config["data"]["text_column"], "/tmp",
    )
    model = model.from_pretrained(checkpoint).to(device)
    model.eval()

    loader = DataLoader(val_set, batch_size=4, collate_fn=CTCCollator(processor, "audio", "transcription"))
    predictions, references = [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(input_values=batch["input_values"].to(device),
                            attention_mask=batch["attention_mask"].to(device)).logits
            predictions.extend(processor.batch_decode(torch.argmax(logits, dim=-1)))
            references.extend(processor.batch_decode(
                torch.where(batch["labels"] == -100, processor.tokenizer.pad_token_id, batch["labels"]),
                group_tokens=False))
    metrics = compute_metrics(references, predictions)
    print(f"  speaker-holdout WER: {metrics['wer_percent']:.2f}% | CER: {metrics['cer_percent']:.2f}%")
