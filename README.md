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
**Cross-domain (FLEURS):** MMS-300M, Whisper Small, Whisper Tiny (for 8 languages with FLEURS)

## Kaggle Setup

1. Upload this repository as a Kaggle Dataset named `waxal-asr-code`.
2. In your Kaggle notebook, add the dataset to get the code at `/kaggle/input/waxal-asr-code/asr_waxal`.
3. Add the code directory to sys.path:
   ```python
   import sys
   sys.path.insert(0, "/kaggle/input/waxal-asr-code/asr_waxal")
   ```
4. Run `kaggle/setup.sh` (or the equivalent pip install cell) to install dependencies.
5. Set environment variables:
   ```python
   import os
   os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
   os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf_cache"
   os.environ["WANDB_DISABLED"] = "true"
   ```

## Running Fine-tuning (Kaggle Notebooks)

Open `kaggle/notebooks/finetune_{language}.ipynb` for the language you want to fine-tune.
Each notebook fine-tunes all 3 models (Whisper Tiny, Whisper Small, MMS-300M) in sequence,
runs evaluation, and (for languages with FLEURS) cross-domain evaluation.

## Running Zero-shot Evaluation

Open `kaggle/notebooks/zero_shot_eval_all.ipynb`. This runs all 19 languages × 3 zero-shot
models (57 total evaluations), grouped by model.

## Aggregating Results

After all runs complete, open `kaggle/notebooks/aggregate_results.ipynb` to build the
paper's Table 1 and Table 3 from all metrics.json files.

Alternatively:
```bash
python -m src.report_results --outputs-dir /kaggle/working/outputs
```

## Config Structure

```
configs/
  data/           # 19 language data configs
  experiment/     # 57 finetune + 57 zero_shot + 24 cross_domain = 138 experiment configs
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
4. Early stopping minimizes validation loss.
5. Effective batch size: 8 per device × 4 accumulation steps.
6. All Kaggle paths use `/kaggle/working/` prefix.
