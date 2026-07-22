from __future__ import annotations

import argparse
import tempfile

import numpy as np
import soundfile as sf
import torch

from src.data.clean_filter import clean_dataset
from src.data.load_waxal import load_waxal
from src.evaluate import predict_ctc, predict_whisper
from src.models.mms_wrapper import load_mms_1b
from src.models.whisper_wrapper import load_whisper
from src.utils.config import resolve_experiment, write_resolved_config
from src.utils.logging_setup import setup_logging, write_json
from src.utils.metrics import compute_metrics
from src.utils.seed import seed_everything


OMNILINGUAL_MAX_AUDIO_SEC = 40


def predict_omnilingual(dataset, config):
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    pipeline = ASRInferencePipeline(model_card=config["model"]["model_card"])
    predictions, references = [], []
    with tempfile.TemporaryDirectory() as directory:
        for index, example in enumerate(dataset):
            path = f"{directory}/{index}.wav"
            audio = example[config["data"]["audio_column"]]
            # The pipeline hard-rejects audio longer than 40s (no chunking);
            # WAXAL's conversational utterances have no max-duration filter,
            # so truncate rather than crash on the (rare) long tail.
            max_samples = OMNILINGUAL_MAX_AUDIO_SEC * audio["sampling_rate"]
            array = audio["array"][:max_samples]
            sf.write(path, array, audio["sampling_rate"])
            predictions.extend(pipeline.transcribe([path], lang=[config["model"]["language_tag"]], batch_size=1))
            references.append(example[config["data"]["text_column"]])
    return predictions, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = resolve_experiment(arguments.config)
    logger = setup_logging(config["output_dir"])
    write_resolved_config(config, config["output_dir"])
    seed_everything(config["seed"])
    if not config["model"].get("supports_language", True):
        result = {"status": "N/A", "reason": "Model has no native zero-shot support for this language."}
        write_json(config["output_dir"], "metrics.json", result)
        logger.info("N/A: %s", result["reason"])
        return
    raw = load_waxal(config["data"])
    cleaned, stats = clean_dataset(raw, config["data"], logger)
    write_json(config["output_dir"], "data_filter_stats.json", stats)
    test = cleaned[config["data"]["test_split"]]
    family = config["model"]["family"]
    if family == "whisper":
        model, processor = load_whisper(config["model"], config["data"]["language"])
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        predictions, references = predict_whisper(model, processor, test, config)
    elif family == "mms_1b":
        model, processor = load_mms_1b(config["model"], config["data"]["iso_code"])
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        predictions, references = predict_ctc(model, processor, test, config)
    elif family == "omnilingual":
        predictions, references = predict_omnilingual(test, config)
    else:
        raise ValueError(f"Unknown zero-shot family: {family}")
    metrics = compute_metrics(references, predictions)
    if metrics["wer_over_100"]:
        logger.warning("WER exceeds 100%%; retaining it as valid insertion-heavy output.")
    write_json(config["output_dir"], "metrics.json", metrics)
    logger.info("WER %.2f%% | CER %.2f%%", metrics["wer_percent"], metrics["cer_percent"])


if __name__ == "__main__":
    main()
