"""FLEURS evaluation wrapper.

Loads the verified FLEURS config specified in a cross-domain experiment, samples
exactly 300 test examples with the configured seed, and delegates decoding to the
appropriate model-specific evaluator. This intentionally shares WER/CER
normalization with the WAXAL evaluation path.
"""
from __future__ import annotations

import argparse

from datasets import Audio, load_dataset
import torch

from src.evaluate import evaluate_model
from src.models.mms_wrapper import load_mms_300m
from src.models.whisper_wrapper import load_whisper
from src.utils.config import resolve_experiment, write_resolved_config
from src.utils.logging_setup import setup_logging, write_json
from src.utils.metrics import wer_confidence_interval
from src.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint")
    args = parser.parse_args()
    config = resolve_experiment(args.config)
    logger = setup_logging(config["output_dir"])
    write_resolved_config(config, config["output_dir"])
    seed_everything(config["seed"])
    fleurs = load_dataset(config["fleurs_dataset_name"], config["fleurs_config"], split=config["fleurs_split"], cache_dir=config.get("cache_dir"))
    fleurs = fleurs.cast_column(config["audio_column"], Audio(sampling_rate=16000)).shuffle(seed=config["seed"]).select(range(config["sample_size"]))
    # FLEURS uses the same verified `audio` and `transcription` field names.
    config["data"] = {"audio_column": config["audio_column"], "text_column": config["text_column"],
                      "train_split": "train"}
    if config["model"]["family"] == "whisper":
        model, processor = load_whisper(config["model"], config["language"])
        model = model.from_pretrained(args.checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
    elif config["model"]["family"] == "mms_300m":
        # The WAXAL training vocabulary saved with the checkpoint must be reused,
        # rather than deriving a new vocabulary from FLEURS references.
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        processor = AutoProcessor.from_pretrained(args.checkpoint)
        model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError("cross_domain_eval currently accepts fine-tuned Whisper or MMS-300M checkpoints")
    predictions, references = evaluate_model(model, processor, fleurs, config)
    from src.utils.metrics import compute_metrics
    metrics = compute_metrics(references, predictions)
    metrics["wer_95_ci"] = wer_confidence_interval(metrics["wer"], config["sample_size"])
    metrics["wer_95_ci_percent"] = metrics["wer_95_ci"] * 100
    write_json(config["output_dir"], "metrics.json", metrics)
    logger.info("FLEURS WER %.2f%% ± %.2f pp | CER %.2f%%", metrics["wer_percent"], metrics["wer_95_ci_percent"], metrics["cer_percent"])


if __name__ == "__main__":
    main()
