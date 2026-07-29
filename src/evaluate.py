from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CTCCollator, WhisperCollator
from src.data.clean_filter import clean_dataset
from src.data.load_waxal import load_waxal
from src.utils.config import resolve_experiment, write_resolved_config
from src.utils.logging_setup import setup_logging, write_json
from src.utils.metrics import compute_metrics
from src.utils.seed import seed_everything


def predict_whisper(model, processor, dataset, config):
    loader = DataLoader(dataset, batch_size=config["evaluation_batch_size"],
                        collate_fn=WhisperCollator(processor, config["data"]["audio_column"], config["data"]["text_column"]))
    device = next(model.parameters()).device
    predictions, references = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            generated = model.generate(batch["input_features"].to(device), do_sample=False, num_beams=1)
            predictions.extend(processor.batch_decode(generated, skip_special_tokens=True))
            references.extend(processor.tokenizer.batch_decode(torch.where(batch["labels"] == -100, processor.tokenizer.pad_token_id, batch["labels"]), skip_special_tokens=True))
    return predictions, references


def predict_ctc(model, processor, dataset, config):
    loader = DataLoader(dataset, batch_size=config["evaluation_batch_size"],
                        collate_fn=CTCCollator(processor, config["data"]["audio_column"], config["data"]["text_column"]))
    device = next(model.parameters()).device
    predictions, references = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(input_values=batch["input_values"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
            predictions.extend(processor.batch_decode(torch.argmax(logits, dim=-1)))
            references.extend(processor.batch_decode(torch.where(batch["labels"] == -100, processor.tokenizer.pad_token_id, batch["labels"]), group_tokens=False))
    return predictions, references


def evaluate_model(model, processor, dataset, config):
    if config["decoding_strategy"] != "greedy":
        raise NotImplementedError("Only greedy decoding is implemented; beam and lm_fusion are extension points.")
    if config["model"]["family"] == "whisper":
        return predict_whisper(model, processor, dataset, config)
    return predict_ctc(model, processor, dataset, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    arguments = parser.parse_args()
    config = resolve_experiment(arguments.config)
    output_dir = config["output_dir"]
    logger = setup_logging(output_dir)
    write_resolved_config(config, output_dir)
    seed_everything(config["model"]["seed"])
    raw = load_waxal(config["data"])
    cleaned, _ = clean_dataset(raw, config["data"], logger)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config["model"]["family"] == "whisper":
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        processor = AutoProcessor.from_pretrained(arguments.checkpoint)
        model = WhisperForConditionalGeneration.from_pretrained(arguments.checkpoint).to(device)
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
    elif config["model"]["family"] == "mms_1b_finetune":
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        processor = AutoProcessor.from_pretrained(arguments.checkpoint)
        model = Wav2Vec2ForCTC.from_pretrained(arguments.checkpoint).to(device)
    else:
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        processor = AutoProcessor.from_pretrained(arguments.checkpoint)
        model = Wav2Vec2ForCTC.from_pretrained(arguments.checkpoint).to(device)
    predictions, references = evaluate_model(model, processor, cleaned[config["data"]["test_split"]], config)
    metrics = compute_metrics(references, predictions)
    if metrics["wer_over_100"]:
        logger.warning("WER exceeds 100%%; retaining it as a valid insertion-heavy result.")
    write_json(output_dir, "metrics.json", metrics)
    logger.info("WER %.2f%% | CER %.2f%%", metrics["wer_percent"], metrics["cer_percent"])


if __name__ == "__main__":
    main()
