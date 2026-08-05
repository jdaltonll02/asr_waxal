from __future__ import annotations

import argparse

from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainerCallback, TrainingArguments

from src.data.clean_filter import clean_dataset
from src.data.dataset import AugmentedCTCCollator, CTCCollator, WhisperCollator
from src.data.load_multilingual import load_multilingual
from src.data.load_multilingual_with_pseudolabels import load_multilingual_with_pseudolabels
from src.data.load_waxal import load_waxal
from src.data.load_waxal_with_fleurs import load_waxal_with_fleurs
from src.data.speaker_split import apply_speaker_holdout
from src.models.mms_wrapper import load_mms_300m, load_mms_1b_finetune
from src.models.whisper_wrapper import load_whisper
from src.utils.config import resolve_experiment, write_resolved_config
from src.utils.logging_setup import setup_logging, write_json
from src.utils.seed import seed_everything


class AugmentToggleCallback(TrainerCallback):
    """Disables augmentation in the collator during evaluation phases."""
    def __init__(self, collator):
        self._collator = collator

    def on_evaluate(self, args, state, control, **kwargs):
        self._collator.training = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._collator.training = True


def training_arguments(config):
    model = config["model"]
    common = dict(output_dir=config["checkpoint_dir"], learning_rate=float(model["learning_rate"]),
                  per_device_train_batch_size=model["per_device_batch_size"], per_device_eval_batch_size=config["evaluation_batch_size"],
                  gradient_accumulation_steps=model["gradient_accumulation_steps"], num_train_epochs=model["max_epochs"],
                  warmup_steps=model["warmup_steps"], lr_scheduler_type=model["lr_schedule"].replace("_decay", ""),
                  eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps", logging_steps=25,
                  load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                  fp16=config["fp16"], report_to=["wandb"] if config.get("use_wandb") else [], remove_unused_columns=False,
                  save_total_limit=2, seed=model["seed"],
                  gradient_checkpointing=model.get("gradient_checkpointing", False))
    return common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help="Path to a Trainer checkpoint-N directory to resume from (e.g. after an OOM).")
    arguments = parser.parse_args()
    config = resolve_experiment(arguments.config)
    logger = setup_logging(config["output_dir"])
    write_resolved_config(config, config["output_dir"])
    seed_everything(config["model"]["seed"])
    if "hf_dataset_configs" in config["data"]:
        loader = load_multilingual_with_pseudolabels if config["data"].get("pseudo_label_dir") else load_multilingual
    elif config["data"].get("use_fleurs_augmentation"):
        loader = load_waxal_with_fleurs
    else:
        loader = load_waxal
    raw = loader(config["data"])
    if config["data"].get("speaker_holdout_validation"):
        raw = apply_speaker_holdout(raw, config["data"])
        logger.info("Applied speaker-disjoint validation holdout (fraction=%.2f)",
                    config["data"].get("speaker_holdout_fraction", 0.15))
    cleaned, stats = clean_dataset(raw, config["data"], logger)
    write_json(config["output_dir"], "data_filter_stats.json", stats)
    train_set, validation_set = cleaned[config["data"]["train_split"]], cleaned[config["data"]["val_split"]]
    model_config = config["model"]
    if model_config["family"] == "whisper":
        model, processor = load_whisper(model_config, config["data"].get("language"))
        args = Seq2SeqTrainingArguments(**training_arguments(config), predict_with_generate=False)
        collator = WhisperCollator(processor, config["data"]["audio_column"], config["data"]["text_column"])
        trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_set, eval_dataset=validation_set,
                                 data_collator=collator, callbacks=[EarlyStoppingCallback(model_config["early_stopping_patience"])])
    elif model_config["family"] == "mms_1b_finetune":
        model, processor = load_mms_1b_finetune(model_config, config["data"]["iso_code"],
                                                 train_set, config["data"]["text_column"], config["output_dir"])
        args = TrainingArguments(**training_arguments(config))
        collator = CTCCollator(processor, config["data"]["audio_column"], config["data"]["text_column"])
        trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=validation_set,
                          data_collator=collator, callbacks=[EarlyStoppingCallback(model_config["early_stopping_patience"])])
    else:
        model, processor = load_mms_300m(model_config, train_set, config["data"]["text_column"], config["output_dir"])
        args = TrainingArguments(**training_arguments(config))
        aug_kwargs = model_config.get("augment_kwargs", {})
        if model_config.get("augment"):
            collator = AugmentedCTCCollator(processor, config["data"]["audio_column"], config["data"]["text_column"], aug_kwargs=aug_kwargs)
            callbacks = [EarlyStoppingCallback(model_config["early_stopping_patience"]), AugmentToggleCallback(collator)]
        else:
            collator = CTCCollator(processor, config["data"]["audio_column"], config["data"]["text_column"])
            callbacks = [EarlyStoppingCallback(model_config["early_stopping_patience"])]
        trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=validation_set,
                          data_collator=collator,
                          callbacks=callbacks)
    trainer.train(resume_from_checkpoint=arguments.resume_from_checkpoint)
    trainer.save_model(config["checkpoint_dir"] + "/best_model")
    processor.save_pretrained(config["checkpoint_dir"] + "/best_model")
    logger.info("Training complete. Evaluate with src.evaluate and %s/best_model", config["checkpoint_dir"])


if __name__ == "__main__":
    main()
