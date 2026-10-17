from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainerCallback,
                          TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

from src.data.clean_filter import DURATION_COLUMN, clean_dataset
from src.data.dataset import CTCCollator, WhisperCollator
from src.data.load_waxal import load_waxal
from src.models.mms_wrapper import load_mms_300m
from src.models.whisper_wrapper import load_whisper
from src.utils.config import resolve_experiment, write_resolved_config
from src.utils.hub import HubMirror, HubSyncCallback, ProgressCallback, has_complete_model
from src.utils.logging_setup import setup_logging, write_json
from src.utils.seed import seed_everything


class DeadlineCallback(TrainerCallback):
    """Save and stop cleanly before a hard session limit (e.g. Kaggle's 12 h)."""

    def __init__(self, deadline: float):
        self.deadline = deadline
        self.triggered = False

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() >= self.deadline:
            self.triggered = True
            control.should_save = True
            control.should_training_stop = True
        return control


def gradient_accumulation(model_config) -> int:
    # The paper's effective batch of 32 is held fixed however many GPUs
    # DataParallel spreads per_device_batch_size over (e.g. Kaggle T4 x2).
    per_step = model_config["per_device_batch_size"] * max(1, torch.cuda.device_count())
    return max(1, model_config["effective_batch_size"] // per_step)


def training_arguments(config):
    model = config["model"]
    common = dict(output_dir=config["checkpoint_dir"], learning_rate=float(model["learning_rate"]),
                  per_device_train_batch_size=model["per_device_batch_size"], per_device_eval_batch_size=config["evaluation_batch_size"],
                  gradient_accumulation_steps=gradient_accumulation(model), num_train_epochs=model["max_epochs"],
                  warmup_steps=model["warmup_steps"], lr_scheduler_type=model["lr_schedule"].replace("_decay", ""),
                  eval_strategy="steps", eval_steps=model["eval_steps"], save_strategy="steps", save_steps=model["eval_steps"],
                  logging_strategy="steps", logging_steps=25, disable_tqdm=True,
                  load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                  restore_callback_states_from_checkpoint=True,
                  gradient_checkpointing=model.get("gradient_checkpointing", False),
                  dataloader_num_workers=model.get("dataloader_num_workers", 0),
                  fp16=config["fp16"] and torch.cuda.is_available(), report_to=["wandb"] if config.get("use_wandb") else [],
                  remove_unused_columns=False, save_total_limit=2, seed=model["seed"])
    return common


def match_saved_batch_size(checkpoint: str, train_batch_size: int, logger) -> None:
    """Make a resumed run keep the configured batch size.

    Trainer.train() replaces its batch size with trainer_state.json's on resume.
    A checkpoint written under DataParallel on 2 GPUs (2 x 2 = 4 per micro-batch)
    and resumed on 1 GPU with doubled accumulation therefore trained at 64, not 32.
    """
    path = os.path.join(checkpoint, "trainer_state.json")
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("train_batch_size") not in (None, train_batch_size):
        logger.warning("Checkpoint micro-batch %s != configured %s; using the configured size.",
                       state["train_batch_size"], train_batch_size)
        state["train_batch_size"] = train_batch_size
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)


def drop_long_audio(dataset, max_seconds, logger, name):
    if not max_seconds:
        return dataset
    kept = dataset.filter(lambda duration: duration <= max_seconds, input_columns=[DURATION_COLUMN])
    logger.info("%s: dropped %d clips longer than %.0fs", name, len(dataset) - len(kept), max_seconds)
    return kept


def drop_overlong_labels(dataset, tokenizer, max_tokens, text_column, logger, name):
    # Whisper's decoder has 448 positions; longer label sequences (common for
    # byte-level BPE on Ge'ez script) would crash the forward pass.
    kept = dataset.filter(lambda texts: [len(ids) <= max_tokens for ids in tokenizer(texts).input_ids],
                          input_columns=[text_column], batched=True)
    logger.info("%s: dropped %d utterances with > %d label tokens", name, len(dataset) - len(kept), max_tokens)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--hub-repo", default=os.environ.get("WAXAL_HUB_REPO"),
                        help="Private HF repo mirroring checkpoints so later sessions can resume")
    parser.add_argument("--deadline", type=float, default=None, help="Unix time at which to save and stop")
    arguments = parser.parse_args()
    config = resolve_experiment(arguments.config)
    logger = setup_logging(config["output_dir"])
    write_resolved_config(config, config["output_dir"])
    best_model_dir = config["checkpoint_dir"] + "/best_model"
    mirror = HubMirror(arguments.hub_repo, config["checkpoint_dir"]) if arguments.hub_repo else None
    if mirror:
        mirror.pull()
    else:
        logger.warning("No --hub-repo: checkpoints live only in this session and cannot be resumed later.")
    if has_complete_model(best_model_dir):
        logger.info("Training already complete: %s exists.", best_model_dir)
        write_json(config["output_dir"], "training_status.json", {"complete": True})
        return
    seed_everything(config["model"]["seed"])
    data_config, model_config = config["data"], config["model"]
    raw = load_waxal(data_config, [data_config["train_split"], data_config["val_split"]])
    cleaned, stats = clean_dataset(raw, data_config, logger)
    write_json(config["output_dir"], "data_filter_stats.json", stats)
    train_set, validation_set = cleaned[data_config["train_split"]], cleaned[data_config["val_split"]]
    max_seconds = model_config.get("max_train_duration_sec")
    train_set = drop_long_audio(train_set, max_seconds, logger, "train")
    validation_set = drop_long_audio(validation_set, max_seconds, logger, "validation")
    callbacks = [EarlyStoppingCallback(model_config["early_stopping_patience"])]
    deadline = DeadlineCallback(arguments.deadline) if arguments.deadline else None
    if deadline:
        callbacks.append(deadline)
    if mirror:
        callbacks.append(HubSyncCallback(mirror))
    callbacks.append(ProgressCallback(mirror))
    if model_config["family"] == "whisper":
        model, processor = load_whisper(model_config)
        # The decoding KV cache is useless in training, and DataParallel gathers
        # it from every GPU onto GPU 0 (Whisper Small OOMed on a T4).
        model.config.use_cache = False
        max_tokens = model.config.max_target_positions
        train_set = drop_overlong_labels(train_set, processor.tokenizer, max_tokens, data_config["text_column"], logger, "train")
        validation_set = drop_overlong_labels(validation_set, processor.tokenizer, max_tokens, data_config["text_column"], logger, "validation")
        args = Seq2SeqTrainingArguments(**training_arguments(config), predict_with_generate=False)
        collator = WhisperCollator(processor, data_config["audio_column"], data_config["text_column"])
        trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_set, eval_dataset=validation_set,
                                 data_collator=collator, callbacks=callbacks)
    else:
        model, processor = load_mms_300m(model_config, train_set, data_config["text_column"], config["output_dir"])
        args = TrainingArguments(**training_arguments(config))
        collator = CTCCollator(processor, data_config["audio_column"], data_config["text_column"])
        trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=validation_set,
                          data_collator=collator, callbacks=callbacks)
    logger.info("Effective batch %d = %d per device x %d GPU(s) x %d accumulation",
                args.per_device_train_batch_size * max(1, args.n_gpu) * args.gradient_accumulation_steps,
                args.per_device_train_batch_size, max(1, args.n_gpu), args.gradient_accumulation_steps)
    resume_from = get_last_checkpoint(config["checkpoint_dir"]) if os.path.isdir(config["checkpoint_dir"]) else None
    if resume_from:
        logger.info("Resuming from %s", resume_from)
        match_saved_batch_size(resume_from, args.train_batch_size, logger)
    trainer.train(resume_from_checkpoint=resume_from)
    if deadline and deadline.triggered:
        logger.warning("Session deadline reached at step %d; rerun this job to resume.", trainer.state.global_step)
        write_json(config["output_dir"], "training_status.json", {"complete": False, "step": trainer.state.global_step})
        return
    trainer.save_model(best_model_dir)
    processor.save_pretrained(best_model_dir)
    write_json(config["output_dir"], "training_status.json",
               {"complete": True, "step": trainer.state.global_step, "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_eval_loss": trainer.state.best_metric})
    if mirror:
        mirror.push("best_model")
        mirror.drop_intermediate_checkpoints()
    logger.info("Training complete. Evaluate with src.evaluate and %s", best_model_dir)


if __name__ == "__main__":
    main()
