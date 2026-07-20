"""FluentNep real-time fluency coach dashboard.

Talks to the FastAPI backend (src/fluentnep/server/main.py) over HTTP.
Three ways to feed it speech, matching the viva demo flow in the master
plan: record from the mic, upload a clip, or play one of the bundled
example clips generated during synthetic data prep.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.environ.get("FLUENTNEP_API_URL", "http://localhost:8000")

TAG_COLORS = {
    "FILLER": "#ff4d4f",       # red
    "REPETITION": "#ffa940",   # orange
    "FALSE_START": "#ffc53d",  # amber
    "REPAIR": "#9254de",       # purple
    "PROLONGATION": "#36cfc9", # teal
}

st.set_page_config(page_title="FluentNep — Real-Time Fluency Coach", page_icon="🎙️", layout="wide")


def render_transcript(tokens: list[str], tags: list[str]) -> str:
    spans = []
    for word, tag in zip(tokens, tags):
        safe_word = html.escape(word)
        if tag == "O":
            spans.append(safe_word)
        else:
            color = TAG_COLORS.get(tag, "#ff4d4f")
            spans.append(
                f'<span style="background:{color}22;color:{color};border-bottom:2px solid {color};'
                f'border-radius:4px;padding:1px 4px;font-weight:600" title="{tag}">{safe_word}</span>'
            )
    return " ".join(spans) if spans else "<i>(no speech detected)</i>"


def render_result(result: dict):
    st.markdown(render_transcript(result["tokens"], result["tags"]), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    score = result["fluency_score"]
    col1.metric("Fluency score", f"{score * 100:.0f}%")
    col2.metric("Words", result["n_words"])
    col3.metric("Disfluencies", result["n_disfluent_words"])
    st.progress(score)

    if result["disfluencies"]:
        st.caption("Detected disfluencies")
        st.dataframe(
            [{"word": d["word"], "type": d["tag"], "position": d["index"]} for d in result["disfluencies"]],
            hide_index=True,
            use_container_width=True,
        )

    legend = "  ".join(
        f'<span style="color:{c};font-weight:700">■</span> {t}' for t, c in TAG_COLORS.items()
    )
    st.markdown(f"<small>{legend}</small>", unsafe_allow_html=True)


def call_api(endpoint: str, **kwargs) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}{endpoint}", timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach the FluentNep API at {API_URL}. Start it with:\n\n"
                  f"`uvicorn fluentnep.server.main:app --reload`")
        return None
    except Exception as e:
        st.error(f"Inference failed: {e}")
        return None


def health_badge():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        info = r.json()
        st.sidebar.success(f"API online · device={info.get('device')}")
    except Exception:
        st.sidebar.error("API offline")


st.title("🎙️ FluentNep")
st.caption("Real-time disfluency detection for code-mixed Nepali-English speech")

with st.sidebar:
    st.header("Status")
    health_badge()
    st.markdown("---")
    st.markdown(
        "**Legend**\n\n"
        "- 🔴 Filler (uhh, matlab, like)\n"
        "- 🟠 Repetition\n"
        "- 🟡 False start\n"
        "- 🟣 Repair\n"
        "- 🔵 Prolongation"
    )
    st.markdown("---")
    if "history" not in st.session_state:
        st.session_state.history = []
    st.metric("Session utterances", len(st.session_state.history))
    if st.session_state.history:
        avg = sum(h["fluency_score"] for h in st.session_state.history) / len(st.session_state.history)
        st.metric("Session avg fluency", f"{avg * 100:.0f}%")

tab_mic, tab_upload, tab_examples, tab_text = st.tabs(
    ["🎤 Record", "📁 Upload clip", "▶️ Example clips", "⌨️ Type text"]
)

with tab_mic:
    st.write("Record a short utterance (natural code-mixed Nepali-English speech works best).")
    audio = st.audio_input("Record")
    if audio is not None:
        with st.spinner("Analyzing..."):
            result = call_api("/infer/audio", files={"file": ("clip.wav", audio.getvalue(), "audio/wav")})
        if result:
            render_result(result)
            st.session_state.history.append(result)

with tab_upload:
    uploaded = st.file_uploader("Upload a WAV/MP3 clip", type=["wav", "mp3"])
    if uploaded is not None:
        with st.spinner("Analyzing..."):
            result = call_api("/infer/audio", files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
        if result:
            render_result(result)
            st.session_state.history.append(result)

with tab_examples:
    manifest_path = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "audio_manifest.jsonl"
    if manifest_path.exists():
        entries = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
        disfluent_examples = [e for e in entries if e["is_disfluent"]][:20]
        options = {f'{e["id"]} — "{e["text"][:60]}"': e for e in disfluent_examples}
        if options:
            choice = st.selectbox("Pick a pre-loaded synthetic clip", list(options.keys()))
            entry = options[choice]
            st.audio(entry["wav_path"])
            if st.button("Analyze this clip"):
                with st.spinner("Analyzing..."):
                    with open(entry["wav_path"], "rb") as f:
                        result = call_api("/infer/audio", files={"file": ("clip.wav", f.read(), "audio/wav")})
                if result:
                    st.caption(f"Ground-truth text: {entry['text']}")
                    render_result(result)
                    st.session_state.history.append(result)
        else:
            st.info("No disfluent example clips found in the manifest yet.")
    else:
        st.info("No synthetic audio manifest found. Run `python scripts/generate_data.py` first.")

with tab_text:
    st.write("Bypass the audio model and tag typed code-mixed text directly (useful to sanity-check the tagger).")
    text = st.text_input("Type a sentence", "ma uhh school jaana chahanchhu")
    if st.button("Tag text"):
        result = call_api("/infer/text", json={"text": text})
        if result:
            render_result(result)
            st.session_state.history.append(result)

if st.session_state.history:
    st.markdown("---")
    st.subheader("Session history")
    st.dataframe(
        [
            {"transcript": h["transcript"], "fluency_score": h["fluency_score"], "disfluencies": h["n_disfluent_words"]}
            for h in st.session_state.history
        ],
        use_container_width=True,
        hide_index=True,
    )
