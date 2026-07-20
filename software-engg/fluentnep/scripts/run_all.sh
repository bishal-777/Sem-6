#!/usr/bin/env bash
# End-to-end pipeline: generate synthetic data, train both models, then
# print how to launch the demo. Run from the fluentnep/ project root.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== 1/4: Generating synthetic dataset ==="
python scripts/generate_data.py --n-text "${N_TEXT:-800}" --n-audio "${N_AUDIO:-400}"

echo "=== 2/4: Training DisfluencyTagger (text-only, fast) ==="
python -m fluentnep.training.train_disfluency_tagger --epochs "${TAGGER_EPOCHS:-15}"

echo "=== 3/4: Training AudioEncoder (CTC, uses GPU if available) ==="
python -m fluentnep.training.train_audio_encoder --epochs "${AUDIO_EPOCHS:-30}"

echo "=== 4/4: Done. Launch the demo with: ==="
echo "  Terminal 1: uvicorn fluentnep.server.main:app --reload --app-dir src"
echo "  Terminal 2: streamlit run app/dashboard.py"
