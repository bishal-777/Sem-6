# FluentNep

A Nepali code-mixed speech disfluency detector and real-time fluency coach —
built as a minor project at ACEM (IOE, TU).

Given a Nepali-English code-mixed utterance, FluentNep transcribes it and
flags disfluencies (fillers, repetitions, false starts, repairs,
prolongations) word by word, live, with a fluency score. It's a from-scratch
speech pipeline: a synthetic disfluency generator, a CNN+Transformer+CTC
audio encoder, a Transformer disfluency tagger, a FastAPI/WebSocket backend,
and a Streamlit dashboard.

## Status

This is a complete, working prototype, trained and verified end to end —
not a system trained on large-scale real speech data. Everything below was
actually run, not just implemented:

| Component | Result |
|---|---|
| Synthetic disfluency generator | 600 code-mixed sentences, 5 disfluency types, all label-aligned |
| Synthesized audio (gTTS) | 300 clips |
| DisfluencyTagger (Transformer, from scratch) | macro F1 = **0.70** on held-out data |
| AudioEncoder (CNN+Transformer+CTC, from scratch) | WER = **43.2%** on held-out data |
| Live demo | REST API, WebSocket streaming, and dashboard all verified working |

The audio side is trained on TTS-synthesized speech, not real spontaneous
Nepali recordings, so it won't transcribe a real microphone accurately —
that's the main gap between this and something you'd actually ship. See
[Ideas for extending this](#ideas-for-extending-this).

Two real bugs surfaced and got fixed while building this (both noted inline
in the code, not swept under the rug): a multi-word-token bug that would
have silently broken label alignment, and a CNN pooling choice that
downsampled audio too aggressively for character-level CTC to align at all
— every training sample's loss was silently zeroing out until that was
fixed. Training curves are in `logs/`.

## Architecture

```
mic / upload → VAD + MFCC (audio/features.py)
             → AudioEncoder: CNN → Transformer → CTC   (models/audio_encoder.py)
             → greedy CTC decode → transcript
             → DisfluencyTagger: Embedding → Transformer → per-word tags (models/disfluency_tagger.py)
             → fluency score = 1 − disfluent_words / total_words
             → FastAPI WebSocket (server/main.py) → Streamlit dashboard (app/dashboard.py)
```

## Setup

```bash
cd fluentnep
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu wheel
pip install -e .
```

## Run the whole pipeline

```bash
./scripts/run_all.sh
```

Or step by step:

```bash
# 1. Generate synthetic text + audio dataset with aligned disfluency labels
python scripts/generate_data.py --n-text 800 --n-audio 400

# 2. Train the disfluency tagger (fast, text-only, CPU is fine)
python -m fluentnep.training.train_disfluency_tagger --epochs 15

# 3. Train the audio encoder (CTC, benefits from GPU)
python -m fluentnep.training.train_audio_encoder --epochs 80
```

Checkpoints land in `checkpoints/`, the dataset in `data/synthetic/`.

## Run the demo

```bash
# Terminal 1 — API
uvicorn fluentnep.server.main:app --reload --app-dir src

# Terminal 2 — dashboard
streamlit run app/dashboard.py
```

Open the Streamlit URL. Four ways to feed it speech:
- **Record**: browser mic recording, analyzed on stop
- **Upload clip**: any wav/mp3
- **Example clips**: pre-loaded synthetic disfluent samples from the generated corpus (best way to see it work correctly, since the model has actually seen this kind of audio)
- **Type text**: bypasses the audio model, tags typed text directly (useful for sanity-checking the tagger alone)

For a literal streaming mic → live terminal demo, no browser needed:

```bash
python scripts/mic_stream_client.py
```

## Project layout

```
src/fluentnep/
  audio/features.py            MFCC / VAD / resampling
  synth/
    vocab.py                   Nepali-English word lists + fillers
    text_generator.py          Template-based clean sentence generation
    disfluency_generator.py    5-type disfluency injector
    tts_generator.py           gTTS-based synthetic audio generation
  data/
    char_vocab.py word_vocab.py datasets.py
  models/
    positional_encoding.py audio_encoder.py disfluency_tagger.py
  training/
    train_audio_encoder.py train_disfluency_tagger.py
  inference/pipeline.py        End-to-end audio → transcript → tags → score
  server/main.py                FastAPI REST + WebSocket
app/dashboard.py                 Streamlit dashboard
scripts/
  generate_data.py run_all.sh mic_stream_client.py plot_spectrogram.py
tests/                            pytest unit tests for generators + features
```

## Testing

```bash
pytest tests/ -v
```

## Ideas for extending this

1. Swap the template-generated text for real scraped Nepali-English text
   (YouTube auto-captions, forum posts, anything code-mixed) for more
   linguistic variety.
2. Mix in real speech — Mozilla Common Voice Nepali and OpenSLR-54 are both
   free — instead of/alongside TTS audio. This is the single biggest lever
   on WER: TTS speech is far cleaner and less varied than how people
   actually talk.
3. Scale up the synthetic corpus (thousands of samples instead of
   hundreds) once real data is in the mix, and train longer.
4. Try a Nepali-capable neural TTS (e.g. Coqui TTS) for more linguistically
   accurate synthetic audio in the meantime.
5. None of this requires touching the models, training loops, API, or
   dashboard — they're already built against this exact data shape.

## Disfluency taxonomy

| Tag | Meaning | Example |
|---|---|---|
| `FILLER` | uhh, matlab, like, basically... | "ma **uhh** school jaanchhu" |
| `REPETITION` | word repeated 1-2 times | "tapai **tapai** kaha januhunchha" |
| `FALSE_START` | word abandoned mid-way, restarted | "ma **kath-** kathmandu bata" |
| `REPAIR` | wrong word/span, then corrected | "sau rupiya **matlab** ek hajar" |
| `PROLONGATION` | stretched vowel sound | "**sooo** what do you think" |
