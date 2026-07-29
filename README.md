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

All artifacts — lightweight (`resolved_config.yaml`, `run.log`,
`data_filter_stats.json`, `metrics.json`, under `output_dir`) and heavy (the HF
model/dataset cache and fine-tuned model checkpoints, under `checkpoint_dir`,
referenced by `--checkpoint`) — are rooted at `/data/hf_cache/jgibson2/waxal`, a
large shared cache mount. `output_dir` originally lived on the much smaller
`/data/user_data/jgibson2` per-user quota; that quota is shared across many
unrelated projects on this account and filled up completely (0 bytes free) even
though these are tiny text files, so it was moved here too.

Create the paper-shaped subset result table after jobs finish:

```bash
python -m src.report_results
```

## Results

All 36 runs (9 finetune + 9 evaluate + 9 zero-shot + 9 cross-domain, across the 3
languages) are complete. Cells are WER% / CER%; `N/A` means the zero-shot model has
no native support for that language.

| Language | Omni. (ZS) | MMS-1B (ZS) | Whisper L (ZS) | MMS-300M (FT) | Whisp. S (FT) | Whisp. T (FT) |
|---|---|---|---|---|---|---|
| Lingala | 37.9 / 14.7 | 34.4 / 14.1 | 222.5 / 164.3 | 38.5 / 16.0 | 44.5 / 24.2 | 73.9 / 49.4 |
| Shona | 34.2 / 6.3 | 47.3 / 11.4 | 374.0 / 191.8 | 26.4 / 5.4 | 43.9 / 23.1 | 43.8 / 20.9 |
| Luganda | 47.9 / 11.7 | 44.3 / 10.7 | N/A | 16.3 / 4.4 | 64.1 / 42.8 | 80.3 / 62.2 |

This reproduces the paper's central claim on this 3-language subset: fine-tuned
edge models (MMS-300M especially) substantially outperform zero-shot foundation
models on WAXAL's spontaneous conversational speech. Whisper Large-v3's zero-shot
WER (222–374%) reflects the runaway-repetition hallucination behavior documented in
`ASR.md`, not a metric error — see `WAXAL-NET.md` for how WER above 100% is
interpreted. MMS-300M's fine-tuned numbers land close to the paper's own reported
values for these three languages (e.g. Luganda 16.3/4.4 vs. the paper's 16.9/3.4).

MMS-300M's fine-tuned column required real debugging to get right: an earlier
version of `freeze_encoder` froze the entire wav2vec2 backbone rather than just the
CNN feature extractor, which left the from-scratch CTC head with no adaptable
signal — it silently collapsed to predicting one near-constant character across
every utterance in all three languages (~99% WER/CER), caught only by inspecting
actual model predictions rather than trusting the metrics table at face value. See
`Architecture.md` for the fix and how it was diagnosed.

## Slurm

Review commands before submission:

```bash
bash slurm/submit_all.sh --dry-run
```

Templates request one GPU on the `general` partition and activate this repo's venv;
adjust time/memory/CPUs if needed. `submit_all.sh` submits `train` jobs first, then
`evaluate` and `cross_domain` jobs with `--dependency=afterok:<train job id>` (both
need that run's checkpoint), and `zero_shot` jobs (no dependency). `evaluate.py` has
no dedicated experiment YAML — it reuses each `finetune_*.yaml` directly via
`template_evaluate.slurm`.

## Assumptions retained in code

1. Omnilingual-1B maps to `facebook/omniASR-CTC-1B` / `omniASR_CTC_1B`.
2. MMS-300M freezes only the pretrained wav2vec2 CNN feature extractor (~4.2M of
   ~315M backbone params) and learns a fresh CTC head/vocabulary from filtered
   training transcriptions. Freezing the full backbone (feature_projection and the
   transformer encoder too) was tried first and left the from-scratch head with no
   adaptable signal — it collapsed to predicting one near-constant output
   regardless of input audio; a higher learning rate alone did not fix this.
3. Both filtering rules run independently on train, validation, and test.
4. Early stopping minimizes validation loss (the paper gives patience, not a
   monitored metric).
5. Initial effective batch size is 8 per device × 4 accumulation steps.

`beam` and `lm_fusion` are reserved config extension points; only greedy
decoding is implemented in this reproduction.
