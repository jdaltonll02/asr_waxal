"""Time MMS-300M training steps on synthetic audio under different settings.

    python kaggle/bench_mms.py            # runs every variant in its own process
    python kaggle/bench_mms.py <variant>  # one variant (used internally)

Isolates GPU throughput from data loading: each sample is 22 s of noise (the
Acholi mean) with a 60-symbol CTC target, batch 2 per device.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

VARIANTS = {
    "current": dict(deterministic=True, checkpointing=True, gpus="0,1"),
    "nondeterministic": dict(deterministic=False, checkpointing=True, gpus="0,1"),
    "nondet_no_ckpt": dict(deterministic=False, checkpointing=False, gpus="0,1"),
    "nondet_1gpu": dict(deterministic=False, checkpointing=True, gpus="0"),
    "current_1gpu": dict(deterministic=True, checkpointing=True, gpus="0"),
}
SECONDS, LABEL_LENGTH, WARMUP_STEPS, TIMED_STEPS = 22, 60, 2, 4


def run_variant(name: str) -> None:
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, Wav2Vec2Config, Wav2Vec2ForCTC

    settings = VARIANTS[name]
    torch.backends.cudnn.deterministic = settings["deterministic"]
    torch.backends.cudnn.benchmark = not settings["deterministic"]
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-300m", vocab_size=40, ignore_mismatched_sizes=True,
                                           pad_token_id=39, ctc_loss_reduction="mean", ctc_zero_infinity=True)
    model.freeze_feature_encoder()

    class Synthetic(torch.utils.data.Dataset):
        def __len__(self):
            return 10_000

        def __getitem__(self, index):
            return {"input_values": torch.randn(SECONDS * 16000), "attention_mask": torch.ones(SECONDS * 16000, dtype=torch.long),
                    "labels": torch.randint(0, 38, (LABEL_LENGTH,))}

    class Timer(TrainerCallback):
        def __init__(self):
            self.marks = []

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.marks.append(time.time())

    timer = Timer()
    args = TrainingArguments(output_dir=f"/tmp/bench_{name}", per_device_train_batch_size=2, gradient_accumulation_steps=2,
                             max_steps=WARMUP_STEPS + TIMED_STEPS, fp16=True, gradient_checkpointing=settings["checkpointing"],
                             report_to=[], save_strategy="no", logging_steps=1000, disable_tqdm=True, dataloader_num_workers=2)
    trainer = Trainer(model=model, args=args, train_dataset=Synthetic(), callbacks=[timer])
    trainer.train()
    per_step = (timer.marks[-1] - timer.marks[WARMUP_STEPS - 1]) / TIMED_STEPS
    samples = 2 * max(1, args.n_gpu) * 2
    print(json.dumps({"variant": name, "sec_per_step": round(per_step, 2), "samples_per_step": samples,
                      "sec_per_sample": round(per_step / samples, 3),
                      "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}), flush=True)


def main() -> None:
    if len(sys.argv) > 1:
        run_variant(sys.argv[1])
        return
    for name, settings in VARIANTS.items():
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": settings["gpus"]}
        result = subprocess.run([sys.executable, __file__, name], env=env, capture_output=True, text=True)
        lines = [line for line in result.stdout.splitlines() if line.startswith('{"variant"')]
        print(lines[-1] if lines else json.dumps({"variant": name, "error": result.stderr[-600:]}), flush=True)


if __name__ == "__main__":
    main()
