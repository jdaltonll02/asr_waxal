from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CTCCollator
from src.data.clean_filter import clean_dataset
from src.data.load_waxal import load_waxal
from src.models.whisper_wrapper import configure_decoding
from src.utils.config import resolve_experiment, write_resolved_config
from src.utils.logging_setup import setup_logging, write_json
from src.utils.metrics import compute_metrics
from src.utils.seed import seed_everything


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def raw_references(dataset, config):
    # References come straight from the transcripts (not decoded label ids), so
    # characters absent from a CTC training vocabulary are not turned into [UNK].
    # Column access does not decode audio; DataLoader order matches (no shuffle).
    return [str(text) for text in dataset[config["data"]["text_column"]]]


WHISPER_WINDOW_SECONDS = 30


def split_evenly(array, sampling_rate: int = 16000, max_seconds: int = WHISPER_WINDOW_SECONDS):
    """Split audio into the fewest equal pieces that each fit Whisper's 30 s window
    (a 31 s clip becomes two 15.5 s pieces rather than 30 s + a 1 s scrap)."""
    size = max_seconds * sampling_rate
    pieces = max(1, -(-len(array) // size))
    step = -(-len(array) // pieces)
    return [array[start:start + step] for start in range(0, len(array), step)] or [array]


def predict_whisper(model, processor, dataset, config):
    """Greedy Whisper decoding. long_audio="truncate" keeps only the first 30 s of a
    clip (the feature extractor's default); "chunk30" transcribes every 30 s piece
    and joins the text, as the transformers ASR pipeline does with chunk_length_s=30."""
    long_audio = config.get("long_audio", "truncate")
    if long_audio not in ("truncate", "chunk30"):
        raise ValueError(f"Unknown long_audio mode: {long_audio}")
    batch_size, audio_column = config["evaluation_batch_size"], config["data"]["audio_column"]
    target = next(model.parameters())
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            audios = dataset[start:start + batch_size][audio_column]
            pieces = [(index, piece) for index, audio in enumerate(audios)
                      for piece in (split_evenly(audio["array"]) if long_audio == "chunk30" else [audio["array"]])]
            texts = [[] for _ in audios]
            for offset in range(0, len(pieces), batch_size):
                group = pieces[offset:offset + batch_size]
                features = processor.feature_extractor([piece for _, piece in group], sampling_rate=16000,
                                                       return_tensors="pt").input_features
                generated = model.generate(features.to(target.device, dtype=target.dtype), do_sample=False, num_beams=1)
                for (index, _), text in zip(group, processor.batch_decode(generated, skip_special_tokens=True)):
                    texts[index].append(text.strip())
            predictions.extend(" ".join(parts) for parts in texts)
    return predictions, raw_references(dataset, config)


def predict_ctc(model, processor, dataset, config):
    loader = DataLoader(dataset, batch_size=config["evaluation_batch_size"],
                        collate_fn=CTCCollator(processor, config["data"]["audio_column"], config["data"]["text_column"]))
    target = next(model.parameters())
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(input_values=batch["input_values"].to(target.device, dtype=target.dtype),
                           attention_mask=batch["attention_mask"].to(target.device)).logits
            predictions.extend(processor.batch_decode(torch.argmax(logits, dim=-1)))
    return predictions, raw_references(dataset, config)


def evaluate_model(model, processor, dataset, config):
    if config["decoding_strategy"] != "greedy":
        raise NotImplementedError("Only greedy decoding is implemented; beam and lm_fusion are extension points.")
    if config["model"]["family"] == "whisper":
        return predict_whisper(model, processor, dataset, config)
    return predict_ctc(model, processor, dataset, config)


def load_finetuned(family: str, checkpoint: str):
    """Load a fine-tuned model with the processor (incl. MMS vocabulary) saved beside it."""
    if family == "whisper":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        processor = WhisperProcessor.from_pretrained(checkpoint)
        model = configure_decoding(WhisperForConditionalGeneration.from_pretrained(checkpoint))
        model.config.use_cache = True  # disabled for training; needed for fast generation
    elif family == "mms_300m":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(checkpoint)
        model = Wav2Vec2ForCTC.from_pretrained(checkpoint)
    else:
        raise ValueError(f"Unsupported fine-tuned family: {family}")
    return model.to(device()), processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--long-audio", choices=("truncate", "chunk30"), help="Whisper only; overrides the config")
    parser.add_argument("--output-dir", help="Overrides the config's output_dir")
    arguments = parser.parse_args()
    config = resolve_experiment(arguments.config)
    if arguments.long_audio:
        config["long_audio"] = arguments.long_audio
    output_dir = config["output_dir"] = arguments.output_dir or config["output_dir"]
    logger = setup_logging(output_dir)
    write_resolved_config(config, output_dir)
    seed_everything(config["model"]["seed"])
    test_split = config["data"]["test_split"]
    raw = load_waxal(config["data"], [test_split])
    cleaned, stats = clean_dataset(raw, config["data"], logger)
    write_json(output_dir, "test_filter_stats.json", stats)
    model, processor = load_finetuned(config["model"]["family"], arguments.checkpoint)
    predictions, references = evaluate_model(model, processor, cleaned[test_split], config)
    metrics = compute_metrics(references, predictions)
    metrics["num_utterances"] = len(references)
    if metrics["wer_over_100"]:
        logger.warning("WER exceeds 100%%; retaining it as a valid insertion-heavy result.")
    write_json(output_dir, "metrics.json", metrics)
    write_json(output_dir, "predictions.json", {"references": references, "predictions": predictions})
    logger.info("WER %.2f%% | CER %.2f%%", metrics["wer_percent"], metrics["cer_percent"])


if __name__ == "__main__":
    main()
