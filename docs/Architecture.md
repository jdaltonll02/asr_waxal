# Architecture

This document describes how the code in this repository is organized and how
data flows through it. For the research methodology being reproduced, see
`WAXAL-NET.md`. For general ASR background, see `ASR.md`.

## Directory layout

```
configs/
  data/         one YAML per language (Lingala, Shona, Luganda)
  models/       one YAML per fine-tuned architecture (Whisper Tiny/Small, MMS-300M)
  zero_shot/    one YAML per zero-shot foundation model (Whisper Large-v3, MMS-1B, Omnilingual-1B)
  experiment/   one YAML per (stage, language, model) run; the actual CLI entry point
src/
  data/         dataset loading, filtering, batch collation
  models/       thin wrappers around HF `transformers` model classes
  utils/        config resolution, logging, metrics, seeding
  train.py            fine-tunes an edge model on WAXAL
  evaluate.py         evaluates a fine-tuned checkpoint on the WAXAL test split
  zero_shot_eval.py   evaluates a foundation model on WAXAL with no fine-tuning
  cross_domain_eval.py  evaluates a fine-tuned checkpoint on FLEURS (out-of-domain)
  report_results.py  assembles the paper-shaped results table from metrics.json files
slurm/          sbatch templates + a driver script that submits every experiment config
```

Every script is invoked the same way: `python -m src.<script> --config <experiment.yaml>`
(plus `--checkpoint <path>` for `evaluate.py` and `cross_domain_eval.py`, since those
operate on an already-trained model rather than training one).

## Config system

Three YAML files compose into one resolved config per run:

- `data_config` — language identity, HF dataset config name, filter thresholds, cache
  location, FLEURS config id.
- `model_config` — architecture family, checkpoint id, hyperparameters (fine-tuned) or
  just `family`/`checkpoint`/`decoding_strategy` (zero-shot).
- The experiment YAML itself — `stage`, `output_dir`, and anything experiment-specific
  (batch size, seed, `use_wandb`, and for zero-shot Whisper Large-v3 the per-language
  `supports_language` override).

`src/utils/config.py:resolve_experiment` reads all three, deep-copies the experiment
dict, nests `data` and `model` under it, applies any per-experiment overrides onto the
model dict (`supports_language`, `language_tag` — this is how one shared
`whisper_large_v3.yaml` model card can still express "this model has no native support
for Luganda" without a per-language copy of the model YAML), and expands
`$VARS`/`~` in every string value. The fully resolved dict is what every script
actually operates on, and `write_resolved_config` snapshots it as
`resolved_config.yaml` next to that run's other outputs — this is the source of truth
for "what actually ran," independent of any later edits to the YAML files themselves.

## Data pipeline

1. **`src/data/load_waxal.py`** — `datasets.load_dataset("google/WaxalNLP", <config>)`,
   then casts the audio column to 16 kHz. Returns the dataset's native `train` /
   `validation` / `test` splits unmodified (the `unlabeled` split is never touched).
2. **`src/data/clean_filter.py`** — applies the paper's two filters to every split
   independently:
   - duration ≥ `min_duration_sec` (1.5s)
   - `words / duration_sec ≤ max_words_per_sec` (4.0) — catches reference-text
     artifacts where the transcript implies an impossible speech rate (see
     `WAXAL-NET.md` for why this matters).
   `clean_dataset` logs utterance-count/hour before and after per split and returns
   that as a stats dict, written to `data_filter_stats.json`.
3. **Collation** (`src/data/dataset.py`) — turns a list of examples into a model-ready
   batch:
   - `WhisperCollator` — feature-extracts audio to log-mel spectrograms and tokenizes
     text, masking padding in the labels with `-100` so it's ignored by the loss.
   - `CTCCollator` — feature-extracts raw waveform (with an attention mask, since
     wav2vec2 needs to know which frames are padding) and character-tokenizes text
     the same way, masking pad tokens with `-100`.

## Model wrappers (`src/models/`)

- **`whisper_wrapper.load_whisper`** — loads a `WhisperForConditionalGeneration` +
  `WhisperProcessor` from a checkpoint id, clears `forced_decoder_ids`/`suppress_tokens`
  so generation isn't locked to a language token Whisper doesn't have for these
  languages.
- **`mms_wrapper.load_mms_300m`** — the interesting one. The base `facebook/mms-300m`
  checkpoint is *acoustic-only*: it has no text head at all. `build_vocab` derives a
  fresh character-level vocabulary directly from the (filtered) WAXAL training
  transcripts, writes it as `vocab.json`, and a `Wav2Vec2CTCTokenizer` /
  `Wav2Vec2ForCTC` pair is constructed around it (`ignore_mismatched_sizes=True` since
  the pretrained checkpoint has no classification head to match sizes with). If
  `freeze_encoder` is set, every `model.wav2vec2.*` parameter has `requires_grad`
  turned off, so training only touches the newly-initialized CTC head — this is why
  the paper describes MMS-300M's fine-tuned performance as "a direct measure of what
  distilling on WAXAL alone through full supervision can induce."
- **`mms_wrapper.load_mms_1b`** — the zero-shot counterpart. `facebook/mms-1b-all`
  ships with per-language adapters; `set_target_lang` + `load_adapter` switch the
  frozen backbone to the target language's existing (not fine-tuned) head.

## The four pipeline stages

| Script | Input | Output | Purpose |
|---|---|---|---|
| `train.py` | WAXAL train+val | `checkpoint_dir/best_model`, logs | fine-tune an edge model |
| `evaluate.py` | WAXAL test + `--checkpoint` | `output_dir/metrics.json` | in-domain WER/CER of a fine-tuned model |
| `zero_shot_eval.py` | WAXAL test | `output_dir/metrics.json` | in-domain WER/CER of a foundation model, no training |
| `cross_domain_eval.py` | FLEURS test (300-sample) + `--checkpoint` | `output_dir/metrics.json` | out-of-domain generalization of a fine-tuned model |

`train.py` builds a `Seq2SeqTrainer` (Whisper) or `Trainer` (MMS-300M) with
`load_best_model_at_end=True` / `metric_for_best_model="eval_loss"` /
`EarlyStoppingCallback(patience=3)`, trains, then saves the best checkpoint. Every
other script loads a checkpoint (or a zero-shot model straight from the Hub), runs a
plain `DataLoader` + `model.generate` (Whisper) or `model(...).logits` (CTC) loop with
no gradient computation, decodes, and calls `compute_metrics`.

`zero_shot_eval.py` additionally handles two model-specific realities that don't apply
to the fine-tuned path:
- **No native language support** — if a zero-shot model's config sets
  `supports_language: false` for a given language (e.g. Whisper Large-v3 on Luganda),
  the script short-circuits immediately and writes `{"status": "N/A", ...}` rather
  than attempting a meaningless forward pass.
- **Omnilingual-1B's 40-second hard cap** — the `omnilingual_asr` inference pipeline
  rejects any single utterance longer than 40 seconds outright with no chunking
  support. Since WAXAL's cleaning filters only enforce a *minimum* duration,
  `predict_omnilingual` truncates each clip to 40s before writing it to the temp
  `.wav` file the pipeline reads from — the same "truncate rather than crash on a
  fixed context window" pattern Whisper already needs (see `ASR.md`).

`cross_domain_eval.py` is a near-mirror of `evaluate.py`, but pulls from FLEURS
instead of WAXAL, samples exactly `sample_size` (300) examples with a fixed seed, and
additionally reports a Wald confidence interval (`wer_confidence_interval` in
`src/utils/metrics.py`) since 300 samples is a much smaller evaluation set than
WAXAL's full test split. For MMS-300M specifically, the checkpoint's own saved
processor/vocab is reloaded (`AutoProcessor.from_pretrained(checkpoint)`) rather than
rebuilding a vocabulary from FLEURS references — the point is to measure the
WAXAL-trained vocabulary's ability to transcribe unfamiliar audio, not to give the
model a vocabulary tailored to the new test set.

## Metrics (`src/utils/metrics.py`)

`normalize_text` applies NFC Unicode normalization, lowercasing, and punctuation
stripping (Unicode letters, combining marks, and digits are preserved, since several
of these languages use diacritics that a naive ASCII-only regex would destroy) before
handing references/hypotheses to `jiwer.wer`/`jiwer.cer`. WER above 100% is not
treated as an error — it's retained and logged as a legitimate insertion-heavy result,
since autoregressive hallucination loops can produce more predicted words than there
were reference words at all.

## Orchestration (`slurm/`)

`submit_all.sh` submits every `configs/experiment/*.yaml` via the matching sbatch
template (`train` → `template_train.slurm`, `zero_shot` → `template_zero_shot_eval.slurm`,
`cross_domain` → `template_cross_domain_eval.slurm`). It submits in two phases —
all `finetune_*` configs first, capturing each one's Slurm job ID — because
`cross_domain_*` jobs evaluate a checkpoint that only exists once the matching
`finetune_*` job has finished; each cross-domain job is submitted with
`--dependency=afterok:<that job id>` so Slurm itself holds it until the checkpoint is
ready, rather than racing the two stages against each other.

### Storage layout

Two independent roots are used, split by artifact size:

- `/data/user_data/jgibson2/waxal/outputs/<run_name>/` — `resolved_config.yaml`,
  `run.log`, `data_filter_stats.json`, `metrics.json`. All KB-scale.
- `/data/hf_cache/jgibson2/waxal/` — the Hugging Face model/dataset cache
  (`HF_HOME`, set by the Slurm templates) and `checkpoints/<run_name>/best_model`
  (GB-scale fine-tuned weights, referenced by `checkpoint_dir` in each `finetune_*`
  experiment config and passed via `--checkpoint` to `evaluate.py`/
  `cross_domain_eval.py`).

This split exists because the former lives on a small per-user quota shared across
many unrelated projects, while the latter is a much larger cluster-wide cache mount —
routing multi-GB downloads and checkpoints there, and keeping only small
human-readable artifacts on the personal quota, is what keeps the pipeline runnable
at all once real WAXAL-scale audio and model weights are involved.

## Results reporting

`report_results.py` reads `metrics.json` for every `(stage, language, model)`
combination under a results root and renders the paper's language × model WER/CER
table as Markdown. A `—` cell means that run hasn't produced a `metrics.json` yet;
`N/A` means the model was recorded as having no native support for that language.
