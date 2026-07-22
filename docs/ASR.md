# Automatic Speech Recognition: a case study

This document explains core Automatic Speech Recognition (ASR) concepts using the
models, code, and even the bugs encountered in this repository as concrete examples.
For what this specific project is measuring, see `WAXAL-NET.md`; for how the code is
organized, see `Architecture.md`.

## What ASR does

An ASR system maps an audio waveform to text. Every model in this project does that
in one of two fundamentally different ways, and the two ways behave very differently
in practice — differently enough that this repository's own experimental results
(and its own bugs) trace directly back to which family a model belongs to.

## Two architecture families

### CTC (Connectionist Temporal Classification) — MMS-300M, MMS-1B

An encoder (wav2vec2, self-supervised on raw audio) produces one output frame per
fixed time-slice of audio, and a linear head classifies each frame independently
into a character (or blank). CTC loss marginalizes over every possible
alignment between frames and the target text, so training doesn't need
frame-level transcription alignment — only the sentence-level text. Decoding is a
simple greedy argmax per frame, followed by collapsing repeated characters and
removing blanks (`predict_ctc` in `src/evaluate.py`).

Practical consequence: CTC has no notion of "generating past what the audio
contains." A silent/short recording just produces fewer non-blank frames. This is
why MMS-300M is *structurally immune* to the runaway-generation failure mode
described below — there is no autoregressive loop to get stuck in. Its
characteristic failure mode instead is **phonetic approximation**: the model gets
the acoustic structure right but substitutes a similar-sounding character or
mis-places a word boundary (e.g. compound splitting).

In this repo, MMS-300M is a case study in **building a text head from nothing**:
the base `facebook/mms-300m` checkpoint has no output vocabulary at all, so
`src/models/mms_wrapper.py:build_vocab` derives one directly from the training
transcripts (every unique lowercase character, plus `[UNK]`/`[PAD]`/word-delimiter
tokens) before a `Wav2Vec2ForCTC` head is even constructed. Its zero-shot sibling,
MMS-1B, instead ships pre-built per-language adapters (`load_adapter` in the same
file) — the same backbone architecture, but the head already exists rather than
being learned from this project's data.

### Sequence-to-sequence, autoregressive — Whisper (Tiny/Small/Large-v3)

An encoder converts a log-mel spectrogram into a hidden representation; a
transformer *decoder* then generates output tokens one at a time, each conditioned
on all previously generated tokens (`model.generate(...)` in `predict_whisper`).
This lets the model learn a strong language-model prior over plausible text — useful
when the acoustic signal is ambiguous — but it also means the model can, in
principle, keep generating forever.

Practical consequence: when acoustic evidence is weak or the model reaches a point
of high uncertainty, the decoder can fall into **unbounded repetition** — plausible-
sounding tokens generated well past where the actual audio ended. This is a known,
documented failure mode of autoregressive ASR (hallucination loops), not specific
to this codebase, but it's exactly the failure mode that shows up as WER *exceeding
100%* — more predicted words than reference words exist at all. `compute_metrics`
in `src/utils/metrics.py` treats WER > 100% as a valid result rather than an error
for precisely this reason: it's signal, not noise.

### The engineering cost of a fixed input window

Whisper's encoder is architecturally locked to exactly 3000 mel-frames of input
(30 seconds), a direct consequence of its fixed positional embeddings. This
surfaced as a real bug during this project's reproduction: `WhisperCollator`
originally called the feature extractor with `padding=True`, which pads a batch to
its own longest example rather than to the fixed 3000 frames Whisper requires.
Any batch where nothing reached the full 30 seconds — which is most batches, since
WAXAL utterances are typically much shorter — produced under-length features and
crashed with `Whisper expects the mel input features to be of length 3000, but
found 2570`. The fix was simply to stop overriding the feature extractor's own
default (`padding="max_length"`), which already knows to pad/truncate every example
to that fixed length regardless of batch composition.

This turns out to be a general pattern, not a Whisper-specific quirk: **any**
pretrained acoustic model built around a fixed context window will reject or mishandle
audio outside that window. Omnilingual-1B's inference pipeline hard-rejects any
single utterance longer than 40 seconds (`ValueError: Max audio length is capped at
40s`), with no chunking support at all. Since WAXAL's cleaning filters only enforce
a *minimum* duration (Section: Data cleaning, in `WAXAL-NET.md`), the long tail of
genuinely long conversational utterances will eventually hit this. The fix in
`predict_omnilingual` (`src/zero_shot_eval.py`) is the same shape as Whisper's own
constraint: truncate to the model's fixed limit rather than crash. Both cases are a
reminder that "spontaneous conversational speech" (WAXAL's whole premise) routinely
violates the fixed-length assumptions baked into models originally evaluated on
short, scripted utterances.

## Zero-shot vs. fine-tuned: what's actually being compared

"Zero-shot" here means a foundation model is evaluated on WAXAL with no
gradient update at all — whatever multilingual competence it acquired during its
own (much larger, much more generic) pretraining is what it brings to this specific
domain. "Fine-tuned" means the *same architecture family*, but a much smaller
model, has had its weights updated on WAXAL's own training split.

The central empirical claim this project reproduces is that **domain
specialization can beat scale**: a 244M-parameter Whisper Small fine-tuned on
WAXAL's actual conversational, spontaneous, code-switched audio can outperform a
1.5B-parameter Whisper Large-v3 that has never seen this specific acoustic
domain — even though the larger model is nominally "more capable" and supports far
more languages. This isn't a claim that scale never matters: the cross-domain
(FLEURS) evaluation in this same pipeline shows the zero-shot models' advantage
*returning* once the test domain shifts back to something closer to their own
pretraining distribution (scripted read speech). The lesson is that "zero-shot WER
on domain X" and "zero-shot WER on domain Y" are not the same claim about a model's
capability — domain match dominates raw parameter count.

## Metrics: WER and CER

**Word Error Rate** is edit distance (substitutions + insertions + deletions,
divided by reference word count) computed at the word level; **Character Error
Rate** is the same computation at the character level. Both are computed here via
`jiwer`, after a normalization pass (`normalize_text` in `src/utils/metrics.py`)
that NFC-normalizes Unicode, lowercases, and strips punctuation — while
deliberately preserving Unicode letters, combining marks, and digits, since several
African orthographies use diacritics and combining characters that a naive
ASCII-oriented punctuation regex would silently destroy.

CER matters more than WER alone for some languages precisely because of how errors
get *counted*: a script where one character encodes a whole syllable (a
consonant-vowel pair) will have a single character-level substitution counted as an
entire word error under WER, even if every other character in that word was
transcribed correctly. This project's own 3-language subset (Lingala, Shona,
Luganda) is entirely Latin-script Bantu, so this particular effect isn't visible in
this repo's own results — it's a finding from the wider 19-language WAXAL-NET paper
this pipeline reproduces a slice of (see `WAXAL-NET.md`) — but it's a useful general
lesson: **WER and CER can disagree substantially depending on the target script**,
and reporting only WER can misrepresent a model's actual character-level accuracy.

## Data quality as an ASR concern, not just a modeling one

Before any acoustic modeling question can be asked fairly, the reference
transcripts themselves have to be trustworthy. `src/data/clean_filter.py` encodes
two simple but effective heuristics for catching bad references without any manual
review: a duration floor (1.5s — anything shorter is unlikely to contain a usable
utterance) and a speech-rate ceiling (4 words/second — a transcript implying a
physically impossible speaking rate is much more likely to be a mismatched or
truncated reference than an unusually fast speaker). The magnitude of this effect
in the source material is large enough to matter: the paper this project reproduces
reports a single language's headline WER dropping over 60 percentage points after
this filtering — the *un*filtered number was measuring data quality, not the
acoustic model at all.

## Decoding strategy

This pipeline implements **greedy decoding only** for both architecture families:
`model.generate(..., num_beams=1)` for Whisper, plain per-frame argmax for CTC.
Beam search and external language-model fusion (`beam`, `lm_fusion` in the config
schema) are reserved extension points that raise `NotImplementedError` if selected
— they would generally improve accuracy at the cost of more compute per utterance,
but aren't part of what this reproduction measures.
