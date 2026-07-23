# WAXAL-NET: methodology, experiment setup, and pipeline flow

This document explains the research question this repository reproduces, the
experimental design behind it, and how each piece of that design maps onto the
concrete stages implemented in `src/`. It summarizes Olufemi et al., *"WAXAL-NET:
Finetuned Edge ASR Across 19 African Languages"* (arXiv:2606.02375), and describes
what this repository specifically reproduces from it. For the code-level view of how
the pipeline is implemented, see `Architecture.md`.

## The research question

Modern multilingual ASR foundation models (Whisper, MMS, Omnilingual ASR) claim
support for hundreds to thousands of languages, but are rarely evaluated on
*spontaneous* African speech, and are too large for the edge/mobile hardware where
African-language audio is often actually collected. WAXAL-NET asks two linked
questions:

1. Can a **compact model fine-tuned on the target domain** (edge-deployable, 3–40×
   smaller) outperform a **much larger zero-shot foundation model** on spontaneous,
   conversational African speech?
2. Does that specialization generalize, or does it just overfit to WAXAL's specific
   recording conditions — tested by evaluating the same fine-tuned models on FLEURS
   (scripted read speech), an out-of-domain benchmark.

The paper's headline result across its full 19-language sweep: fine-tuned edge models
reach a macro-averaged WER of 38.0%, versus 64.9% for the best zero-shot baseline — a
26.9 percentage-point gap, using models 3–40× smaller. Domain match (WAXAL's
conversational audio vs. FLEURS' read speech), not raw parameter count, turns out to
be the dominant factor in relative performance.

## Corpus

**WAXAL** (Diack et al., 2026) is a large-scale corpus of transcribed,
image-prompted spontaneous speech recorded in participants' natural environments
across 19 African languages — released under CC-BY-4.0 with a pre-defined
train/validation/test split (362,125 / 44,232 / 39,812 utterances in the labeled
subset used for experimentation). Unlike Common Voice or FLEURS, which capture
scripted *read* speech, WAXAL captures real conversational speech: pausing,
code-switching, dialectal variation — the conditions foundation models are weakest
on and least evaluated against.

This repository reproduces the pipeline for **3 of the 19 languages**:

| Language | WAXAL config | FLEURS config | Family |
|---|---|---|---|
| Lingala | `lin_asr` | `ln_cd` | Niger-Congo (Bantu) |
| Shona | `sna_asr` | `sn_zw` | Niger-Congo (Bantu) |
| Luganda | `lug_asr` | `lg_ug` | Niger-Congo (Bantu) |

(See `configs/data/*.yaml`.) Since all three happen to be Bantu languages, this
subset cannot itself reproduce the paper's cross-family findings (e.g. the
CTC-vs-Bantu / autoregressive-vs-Afro-Asiatic split, or the Ge'ez-script CER/WER
discussion) — those are genuinely 19-language-scope results. What this subset does
faithfully reproduce is the core comparison: fine-tuned edge vs. zero-shot
foundation, and in-domain vs. cross-domain generalization, for a representative
slice of the corpus.

## Data cleaning and filtering

Before any model sees WAXAL audio, two heuristics are applied independently to
train, validation, and test (`src/data/clean_filter.py`):

1. **Duration floor** — discard utterances shorter than 1.5 seconds.
2. **Speech-rate ceiling** — discard utterances where the reference transcript
   implies more than 4 words per second of audio. A physically impossible speech
   rate is a strong signal that the reference text itself is corrupted or
   mismatched to its audio, not that the speaker was unusually fast.

The paper reports this mitigation as significant, not cosmetic: Lingala Whisper Tiny
WER improved from 113.5% to 49.0% after filtering — the *un*filtered number was
measuring reference-text artifacts, not acoustic modeling quality. This is why
filtering happens before every stage (train, evaluate, zero-shot, cross-domain) and
why `data_filter_stats.json` records before/after utterance counts and hours for
every run — it's evidence that the comparison being made is a fair one.

## Models under comparison

**Zero-shot foundation models** (no WAXAL fine-tuning):

| Model | Parameters | Architecture |
|---|---|---|
| Whisper Large-v3 | 1.5B | encoder-decoder, autoregressive |
| MMS-1B | 1B | wav2vec2 + CTC, per-language adapters |
| Omnilingual-1B | 1B | multilingual acoustic pretraining + CTC |

**Fine-tuned edge models** (full fine-tuning on the WAXAL train split, except where
noted):

| Model | Parameters | Architecture | Footprint |
|---|---|---|---|
| Whisper Tiny | 39M | encoder-decoder, autoregressive | ~151 MB |
| Whisper Small | 244M | encoder-decoder, autoregressive | ~967 MB |
| MMS-300M | 300M | wav2vec2 + CTC, feature extractor frozen | ~1.2 GB |

MMS-300M is fine-tuned differently from the two Whisper variants: its base
checkpoint has no text-generation head at all, so a character-level CTC vocabulary
is built fresh from the WAXAL training transcripts (see `Architecture.md` /
`build_vocab`), and only the pretrained wav2vec2 CNN feature extractor (~4.2M of
~315M backbone params) is frozen — `feature_projection`, the transformer encoder,
and the new CTC head all train. (An earlier version of this reproduction froze the
*entire* backbone, matching a naive reading of "distilling on WAXAL alone through
full supervision"; that left the from-scratch head with no adaptable signal and it
collapsed to predicting a near-constant output regardless of input audio — a
training bug, not a real result. See `Architecture.md` for how it was diagnosed and
fixed.) This still makes MMS-300M's fine-tuned result a meaningful measure of what
training on WAXAL alone can induce, since there is no prior text-generation
capability to fall back on — it just isn't a *linear-probe-only* measure.

## Training setup

Hyperparameters are held fixed across languages to keep cross-language comparisons
fair (`configs/models/*.yaml`):

- Effective batch size 32 (8 per-device × 4 gradient-accumulation steps)
- Learning rate 1e-4, linear warmup over 500 steps, polynomial decay
- Up to 30 epochs, early stopping with patience 3 on validation loss
- Whisper: full fine-tuning (all weights updated), greedy decoding, repetition
  penalty disabled
- MMS-300M: feature extractor frozen, CTC greedy decoding

## Evaluation

WER and CER are computed with `jiwer` after lowercasing and punctuation removal
(`src/utils/metrics.py`). WER above 100% is retained rather than treated as invalid —
autoregressive hallucination loops can produce more predicted tokens than reference
tokens exist, and that *is* the finding for languages where it happens, not a bug in
the metric.

**Cross-domain (FLEURS) generalization**: each fine-tuned checkpoint is additionally
evaluated on 300 randomly sampled FLEURS test utterances for the language it was
trained on, using a fixed seed for reproducibility. Because 300 samples is a much
smaller test set than WAXAL's own split, results are reported with a Wald confidence
interval (`wer_confidence_interval`) rather than treated as a point estimate on par
with the full WAXAL test evaluation.

## Pipeline flow, stage by stage

```
                    ┌─────────────────────┐
                    │  WAXAL (HF dataset)  │
                    └──────────┬──────────┘
                               │ load_waxal (16 kHz cast)
                               ▼
                    ┌─────────────────────┐
                    │  clean_dataset       │  duration ≥1.5s, words/sec ≤4
                    │  (train/val/test)    │
                    └──────────┬──────────┘
                 ┌─────────────┼──────────────┐
                 ▼             ▼              ▼
           train.py     zero_shot_eval.py   (test split only,
    (train+val split)   (test split only,    reused by evaluate.py)
                          no fine-tuning)
                 │             │
                 ▼             ▼
         checkpoint_dir/  output_dir/
         best_model       metrics.json
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
  evaluate.py         cross_domain_eval.py
  (WAXAL test,        (FLEURS 300-sample,
   in-domain WER/CER)  out-of-domain WER/CER)
       │                   │
       └─────────┬─────────┘
                 ▼
         report_results.py
      (paper-shaped results table)
```

Each experiment YAML under `configs/experiment/` pins one `(stage, language, model)`
combination to this flow — `finetune_*` configs drive `train.py` then
`evaluate.py`, `zero_shot_*` configs drive `zero_shot_eval.py` directly (no training
stage), and `cross_domain_*` configs drive `cross_domain_eval.py` against a
`finetune_*` run's checkpoint. `slurm/submit_all.sh` submits all three stages with
the correct dependency ordering (see `Architecture.md`).

## Assumptions made explicit in this reproduction

Documented in `README.md` and preserved here for context:

1. "Omnilingual-1B" maps to the `facebook/omniASR-CTC-1B` checkpoint / `omniASR_CTC_1B`
   model card, since the paper doesn't pin an exact Hub identifier.
2. MMS-300M freezes only the pretrained wav2vec2 CNN feature extractor and learns a
   fresh CTC head/vocabulary from filtered training transcriptions (the paper
   describes this qualitatively; this repo's exact freezing scope is an
   implementation choice — freezing the full backbone was tried first and produced
   a collapsed, non-functional model; see `Architecture.md`).
3. Both cleaning filters run independently on train, validation, and test — the
   paper doesn't specify per-split filtering behavior explicitly.
4. Early stopping monitors validation loss (the paper specifies patience=3 but not
   which metric is monitored).
5. Initial effective batch size is realized as 8 per device × 4 gradient-accumulation
   steps (the paper specifies the effective batch size of 32 but not this split).

Only greedy decoding is implemented; `beam` and `lm_fusion` are reserved config
extension points that raise `NotImplementedError` if selected.
