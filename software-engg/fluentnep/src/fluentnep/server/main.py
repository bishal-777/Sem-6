"""FastAPI backend: WebSocket endpoint for real-time audio streaming plus a
plain REST endpoint for uploaded audio files / typed text. Both are backed
by the same FluentNepPipeline instance, loaded once at startup.
"""
from __future__ import annotations

import io
import logging
import struct
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fluentnep.config import SAMPLE_RATE
from fluentnep.inference.pipeline import FluentNepPipeline

logger = logging.getLogger("fluentnep.server")

pipeline: FluentNepPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading FluentNep models...")
    pipeline = FluentNepPipeline()
    logger.info("Models loaded on device=%s", pipeline.device)
    yield


app = FastAPI(title="FluentNep API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "device": str(pipeline.device) if pipeline else None}


@app.post("/infer/text")
def infer_text(req: TextRequest):
    result = pipeline.run_on_text(req.text)
    return result.to_dict()


@app.post("/infer/audio")
async def infer_audio(file: UploadFile = File(...)):
    raw = await file.read()
    y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    result = pipeline.run(waveform=y, sr=SAMPLE_RATE)
    return result.to_dict()


CHUNK_DURATION_SEC = 0.5
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    """Client streams raw 16-bit PCM mono 16kHz audio in ~500ms chunks
    (binary frames). We accumulate a rolling buffer and run inference on it
    every chunk, echoing back JSON with the live transcript, per-word
    disfluency tags, and fluency score — the loop the dashboard renders.
    """
    await websocket.accept()
    buffer = np.zeros(0, dtype=np.float32)
    max_buffer_sec = 12  # keep a rolling window so latency stays bounded

    try:
        while True:
            data = await websocket.receive_bytes()
            n_samples = len(data) // 2  # int16 -> 2 bytes/sample
            pcm = struct.unpack(f"<{n_samples}h", data)
            chunk = np.array(pcm, dtype=np.float32) / 32768.0

            buffer = np.concatenate([buffer, chunk])
            max_samples = int(SAMPLE_RATE * max_buffer_sec)
            if len(buffer) > max_samples:
                buffer = buffer[-max_samples:]

            if len(buffer) < SAMPLE_RATE * 0.3:
                continue  # not enough audio yet

            result = pipeline.run(waveform=buffer, sr=SAMPLE_RATE)
            await websocket.send_json(result.to_dict())
    except WebSocketDisconnect:
        logger.info("Client disconnected")
