"""FLEURS evaluation wrapper.

Loads the verified FLEURS config specified in a cross-domain experiment, samples
exactly 300 test examples with the configured seed, and delegates decoding to the
appropriate model-specific evaluator. This intentionally shares WER/CER
normalization with the WAXAL evaluation path.
"""
from __future__ import annotations

import argparse

from datasets import Audio, load_dataset

from src.evaluate import evaluate_model, load_finetuned
from src.utils.config import resolve_experiment, write_resolved_config
from src.utils.logging_setup import setup_logging, write_json
from src.utils.metrics import compute_metrics, wer_confidence_interval
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
    fleurs = load_dataset(config["fleurs_dataset_name"], config["fleurs_config"], split=config["fleurs_split"], cache_dir=config.get("cache_dir"),
                          trust_remote_code=True)  # google/fleurs is a loading-script dataset
    fleurs = fleurs.cast_column(config["audio_column"], Audio(sampling_rate=16000)).shuffle(seed=config["seed"]).select(range(config["sample_size"]))
    # FLEURS uses the same verified `audio` and `transcription` field names.
    config["data"] = {"audio_column": config["audio_column"], "text_column": config["text_column"],
                      "train_split": "train"}
    # Fine-tuned checkpoints carry their own processor; for MMS-300M this reuses
    # the WAXAL training vocabulary rather than deriving one from FLEURS.
    model, processor = load_finetuned(config["model"]["family"], args.checkpoint)
    predictions, references = evaluate_model(model, processor, fleurs, config)
    metrics = compute_metrics(references, predictions)
    metrics["wer_95_ci"] = wer_confidence_interval(metrics["wer"], config["sample_size"])
    metrics["wer_95_ci_percent"] = metrics["wer_95_ci"] * 100
    write_json(config["output_dir"], "metrics.json", metrics)
    logger.info("FLEURS WER %.2f%% ± %.2f pp | CER %.2f%%", metrics["wer_percent"], metrics["wer_95_ci_percent"], metrics["cer_percent"])


if __name__ == "__main__":
    main()
