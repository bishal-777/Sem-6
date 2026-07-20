"""True real-time microphone -> WebSocket streaming client.

The Streamlit dashboard records a full utterance before analyzing it
(simplest reliable path in a browser without extra JS components). This
script instead streams your live microphone straight to the FastAPI
WebSocket endpoint in 500ms chunks and prints live results to the
terminal — the literal "audio input -> ... -> live dashboard" pipeline
from the master plan's system design, running with the terminal as the
dashboard.

Requires a local microphone (won't work in a headless/CI environment).

Usage:
  python scripts/mic_stream_client.py --url ws://localhost:8000/ws/stream
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import numpy as np
import sounddevice as sd
import websockets

SAMPLE_RATE = 16_000
CHUNK_SEC = 0.5
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)


async def stream(url: str):
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(audio_queue.put_nowait, pcm16)

    print(f"Connecting to {url} ...")
    async with websockets.connect(url) as ws:
        print("Connected. Speak into your microphone (Ctrl+C to stop).\n")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=callback,
        ):
            async def sender():
                while True:
                    chunk = await audio_queue.get()
                    await ws.send(chunk)

            async def receiver():
                async for message in ws:
                    data = json.loads(message)
                    fluency = data["fluency_score"] * 100
                    print(f"\r[{fluency:5.1f}% fluent] {data['transcript']:<80}", end="", flush=True)

            await asyncio.gather(sender(), receiver())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws/stream")
    args = parser.parse_args()
    try:
        asyncio.run(stream(args.url))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
