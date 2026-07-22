from __future__ import annotations

import argparse

from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainingArguments

from src.data.clean_filter import clean_dataset
from src.data.dataset import CTCCollator, WhisperCollator
from src.data.load_waxal import load_waxal
from src.models.mms_wrapper import load_mms_300m
from src.models.whisper_wrapper import load_whisper
from src.utils.config import resolve_experiment, write_resolved_config
from src.utils.logging_setup import setup_logging, write_json
from src.utils.seed import seed_everything


def training_arguments(config):
    model = config["model"]
    common = dict(output_dir=config["checkpoint_dir"], learning_rate=float(model["learning_rate"]),
                  per_device_train_batch_size=model["per_device_batch_size"], per_device_eval_batch_size=config["evaluation_batch_size"],
                  gradient_accumulation_steps=model["gradient_accumulation_steps"], num_train_epochs=model["max_epochs"],
                  warmup_steps=model["warmup_steps"], lr_scheduler_type=model["lr_schedule"].replace("_decay", ""),
                  eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps", logging_steps=25,
                  load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                  fp16=config["fp16"], report_to=["wandb"] if config.get("use_wandb") else [], remove_unused_columns=False,
                  save_total_limit=2, seed=model["seed"])
    return common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = resolve_experiment(arguments.config)
    logger = setup_logging(config["output_dir"])
    write_resolved_config(config, config["output_dir"])
    seed_everything(config["model"]["seed"])
    raw = load_waxal(config["data"])
    cleaned, stats = clean_dataset(raw, config["data"], logger)
    write_json(config["output_dir"], "data_filter_stats.json", stats)
    train_set, validation_set = cleaned[config["data"]["train_split"]], cleaned[config["data"]["val_split"]]
    model_config = config["model"]
    if model_config["family"] == "whisper":
        model, processor = load_whisper(model_config, config["data"]["language"])
        args = Seq2SeqTrainingArguments(**training_arguments(config), predict_with_generate=False)
        collator = WhisperCollator(processor, config["data"]["audio_column"], config["data"]["text_column"])
        trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_set, eval_dataset=validation_set,
                                 data_collator=collator, callbacks=[EarlyStoppingCallback(model_config["early_stopping_patience"])])
    else:
        model, processor = load_mms_300m(model_config, train_set, config["data"]["text_column"], config["output_dir"])
        args = TrainingArguments(**training_arguments(config))
        collator = CTCCollator(processor, config["data"]["audio_column"], config["data"]["text_column"])
        trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=validation_set,
                          data_collator=collator, callbacks=[EarlyStoppingCallback(model_config["early_stopping_patience"])])
    trainer.train()
    trainer.save_model(config["checkpoint_dir"] + "/best_model")
    processor.save_pretrained(config["checkpoint_dir"] + "/best_model")
    logger.info("Training complete. Evaluate with src.evaluate and %s/best_model", config["checkpoint_dir"])


if __name__ == "__main__":
    main()
