# WAXAL-NET reproduction: Lingala, Shona, Luganda

Config-driven reproduction of the WAXAL-NET baseline pipeline for WAXAL ASR.
It implements the paper’s 1.5-second duration and 4 words/second filters,
greedy decoding, normalized jiwer WER/CER, and deterministic FLEURS samples.

## Verified data identifiers

| Language | WAXAL config | WAXAL fields | FLEURS config |
|---|---|---|---|
| Lingala | `lin_asr` | `audio`, `transcription` | `ln_cd` |
| Shona | `sna_asr` | `audio`, `transcription` | `sn_zw` |
| Luganda | `lug_asr` | `audio`, `transcription` | `lg_ug` |

The source is the official `google/WaxalNLP` dataset. Its labeled split names
are `train`, `validation`, and `test`; `unlabeled` is intentionally ignored.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m src.train --config configs/experiment/finetune_lingala_whisper_small.yaml
python -m src.evaluate --config configs/experiment/finetune_lingala_whisper_small.yaml \
  --checkpoint /data/hf_cache/jgibson2/waxal/checkpoints/finetune_lingala_whisper_small/best_model
python -m src.zero_shot_eval --config configs/experiment/zero_shot_lingala_mms_1b.yaml
python -m src.cross_domain_eval --config configs/experiment/cross_domain_lingala_whisper_small.yaml \
  --checkpoint /data/hf_cache/jgibson2/waxal/checkpoints/finetune_lingala_whisper_small/best_model
```

Every invocation saves `resolved_config.yaml`, filter counts/hours, local JSON
metrics, and a log under its configured `output_dir`. WER above 100% is logged
as a valid insertion-heavy result. Set `use_wandb: true` in an experiment only
after authenticating W&B; local logging is always retained.

Lightweight artifacts (`resolved_config.yaml`, `run.log`, `data_filter_stats.json`,
`metrics.json`) are rooted at `/data/user_data/jgibson2/waxal/outputs`. Heavy
artifacts — the HF model/dataset cache and fine-tuned model checkpoints
(`checkpoint_dir`, referenced by `--checkpoint`) — are rooted at
`/data/hf_cache/jgibson2/waxal` instead, since the former is a small per-user
quota and the latter is a much larger shared cache mount.

Create the paper-shaped subset result table after jobs finish:

```bash
python -m src.report_results
```

## Slurm

Review commands before submission:

```bash
bash slurm/submit_all.sh --dry-run
```

Templates request one GPU; adjust time, memory, CPUs, and your Babel environment
activation at the marked locations. Do not set a partition until `sinfo` confirms
the appropriate Babel partition. For MMS-300M use an A100/A800 80-GB node; use a
40-GB-equivalent node for Whisper Tiny/Small.

## Assumptions retained in code

1. Omnilingual-1B maps to `facebook/omniASR-CTC-1B` / `omniASR_CTC_1B`.
2. MMS-300M freezes the complete pretrained wav2vec2 backbone and learns a fresh
   CTC head/vocabulary from filtered training transcriptions.
3. Both filtering rules run independently on train, validation, and test.
4. Early stopping minimizes validation loss (the paper gives patience, not a
   monitored metric).
5. Initial effective batch size is 8 per device × 4 accumulation steps.

`beam` and `lm_fusion` are reserved config extension points; only greedy
decoding is implemented in this reproduction.
