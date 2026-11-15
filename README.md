# WAXAL-NET Reproduction: All 19 Languages (Kaggle)

Config-driven reproduction of the WAXAL-NET baseline pipeline for all 19 languages from the paper.
Implements the paper's 1.5-second duration and 4 words/second filters, greedy decoding,
normalized jiwer WER/CER, and deterministic FLEURS samples.

## Language Coverage

| Language | ISO | FLEURS | Script |
|----------|-----|--------|--------|
| Acholi | ach | — | Latin |
| Akan | aka | — | Latin |
| Amharic | amh | am_et | Ge'ez |
| Dagaare | dga | — | Latin |
| Dagbani | dag | — | Latin |
| Ewe | ewe | — | Latin |
| Fula | ful | ff_sn | Latin |
| Ikposo | kpo | — | Latin |
| Lingala | lin | ln_cd | Latin |
| Luganda | lug | lg_ug | Latin |
| Malagasy | mlg | mg_mg | Latin |
| Masaaba | mas | — | Latin |
| Nyankole | nyn | — | Latin |
| Oromo | orm | om_et | Latin |
| Shona | sna | sn_zw | Latin |
| Sidama | sid | — | Latin |
| Soga | sog | — | Latin |
| Tigrinya | tir | ti_et | Ge'ez |
| Wolaytta | wal | — | Latin |

## Models

**Zero-shot:** MMS-1B, Omnilingual-1B, Whisper Large-v3 (supports Amharic, Lingala, Malagasy, Shona only)
**Fine-tuned:** MMS-300M, Whisper Small, Whisper Tiny
**Cross-domain (FLEURS):** MMS-300M, Whisper Small, Whisper Tiny on the paper's 6 overlapping languages
(Amharic, Fula, Lingala, Luganda, Oromo, Shona). Malagasy and Tigrinya configs also exist but are not
scheduled, because the paper does not report them.

## Running on Kaggle

Every job is a generated notebook in `kaggle/kernels/<slug>/`, driven by `kaggle/jobs.py`:

- 57 fine-tuning jobs `waxal-ft-<language>-<model>`, one (language, model) pair each. Each trains, then
  evaluates on the WAXAL test split, then (for the 6 FLEURS languages) on 300 FLEURS utterances.
- 5 zero-shot jobs `waxal-zs-<model>[-a|-b]`, each looping over its languages.

One-time setup:

1. `pip install kaggle` and put your API token in `~/.kaggle/kaggle.json`.
2. Create a Hugging Face **write** token, log in with it (`hf auth login --force`), then run
   `python kaggle/jobs.py push-token`. This checks the token and uploads it as the private Kaggle
   dataset `waxal-hf-token`, which every job attaches. (Kaggle secrets are dropped from versions
   started with `kaggle kernels push`.) Each fine-tuning job mirrors its checkpoints to a public
   `waxal-net-finetune_<language>_<model>` model repo, which lets a job outlive the 12 h session limit.
3. Upload the code (sub-folders are zipped; the CLI skips them otherwise):
   ```bash
   python kaggle/jobs.py push-code -m "describe the change"
   ```

Run jobs (each push starts a GPU T4 x2 run with internet on):

```bash
python kaggle/jobs.py push ft-acholi-whisper-tiny   # smoke test first
python kaggle/jobs.py push ft-acholi ft-luganda      # every job whose slug contains a pattern
python kaggle/jobs.py status ft-acholi
```

Without the token dataset a job still runs, but it cannot resume. If you change training code or
configs, delete that job's `waxal-net-...` Hub repo first; otherwise the rerun reuses the old model.

A job that hits its training deadline (10.5 h into the session) saves, pushes and stops. Push it again
and it resumes from the last checkpoint. A job whose training already finished goes straight to evaluation.

Collect results and compare them with the paper:

```bash
python kaggle/jobs.py collect        # downloads job outputs -> results/waxal_results.md
```

The code is never installed into Kaggle's Python 3.13. `kaggle/setup_env.sh` builds a pinned
Python 3.12 venv with `uv`, because numpy 1.26 and fairseq2 (Omnilingual) have no 3.13 wheels.

## Config Structure

```
configs/
  data/           # 19 language data configs
  experiment/     # 57 finetune + 57 zero_shot + 24 cross_domain experiment configs
  models/         # Finetune model configs (whisper_tiny, whisper_small, mms_300m)
  zero_shot/      # Zero-shot model configs (mms_1b, omnilingual_1b, whisper_large_v3)
```

## Verified Data Identifiers

All languages use `google/WaxalNLP` with `{iso}_asr` as the dataset config.
Split names: `train`, `validation`, `test`. Audio field: `audio`. Text field: `transcription`.

## Assumptions

1. Omnilingual-1B maps to `facebook/omniASR-CTC-1B` / `omniASR_CTC_1B`.
2. MMS-300M freezes only the CNN feature extractor; the transformer backbone adapts with a fresh CTC head.
3. Both filtering rules run independently on train, validation, and test.
4. Early stopping minimizes validation loss, evaluated and checkpointed every 500 steps (all of
   the paper's Table 5 step counts are multiples of 500).
5. Effective batch size 32 = per-device batch × GPUs × accumulation (accumulation is derived at run time).
6. Outputs and checkpoints go under `/kaggle/working/`; the HF cache goes in `/tmp/hf_cache` so it
   does not count against the 20 GB output limit.
7. Only the labelled `train`/`validation`/`test` parquet files are downloaded. Each WAXAL config also
   has a 20-35 GB `unlabeled` split that is never used.
8. Whisper training drops utterances whose labels exceed the decoder's 448 positions (logged per run).
