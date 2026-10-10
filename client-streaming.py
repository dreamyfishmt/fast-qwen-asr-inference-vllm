# Client for streaming ASR - this is for TESTING PURPOSES

import argparse
import asyncio
import json
import os
import sys
import time
import wave
from datetime import datetime
from urllib.parse import urlencode

import websockets

CHUNK_MS = 20


def load_audio(path: str, raw_rate: int):
    """Returns (16-bit mono PCM bytes, sample rate). WAV files must be 16-bit mono; anything else is raw PCM at raw_rate."""
    if path.lower().endswith(".wav"):
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                sys.exit(f"{path}: needs 16-bit mono WAV (convert with: ffmpeg -i in -ac 1 -c:a pcm_s16le out.wav)")
            return w.readframes(w.getnframes()), w.getframerate()
    with open(path, "rb") as f:
        return f.read(), raw_rate


async def sender(ws, pcm: bytes, rate: int, path: str):
    # Handshake / Config
    await ws.send(json.dumps({
        "type": "start",
        "format": "pcm_s16le",
        "sample_rate_hz": rate,
        "channels": 1
    }))

    print(f"Streaming {path} ({rate} Hz)...")
    chunk_bytes = rate * 2 * CHUNK_MS // 1000
    for i in range(0, len(pcm), chunk_bytes):
        await ws.send(pcm[i:i + chunk_bytes])
        await asyncio.sleep(0)  # Yield control to ensure receiver can process messages

    await ws.send(json.dumps({"type": "stop"}))
    print("Finished sending audio.")

async def receiver(ws):
    async for message in ws:
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            evt = json.loads(message)
            msg_type = evt.get('type')
            text = evt.get('text', '')
            lang = evt.get('language', '')

            if msg_type == 'ready':
                print(f"[{timestamp}] [Server Ready]")
            elif msg_type == 'partial':
                # Overwrite line for partial updates to keep clean output
                sys.stdout.write(f"\r[{timestamp}] [Partial] ({lang}): {text}")
                sys.stdout.flush()
            elif msg_type == 'final':
                print(f"\n[{timestamp}] [Final] ({lang}): {text}")
            elif msg_type == 'error':
                print(f"\n[{timestamp}] [Error]: {evt.get('message')}")
            else:
                print(f"\n[{timestamp}] [Unknown]: {evt}")

        except json.JSONDecodeError:
            print(f"\n[Raw]: {message}")

async def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Streaming Client")
    parser.add_argument("-e", "--endpoint", required=True, help="WebSocket Endpoint URL (e.g. ws://localhost:8907/transcribe-streaming)")
    parser.add_argument("-f", "--file", required=True, help="16-bit mono WAV file (any sample rate), or raw PCM (16-bit mono, see -r)")
    parser.add_argument("-r", "--rate", type=int, default=16000, help="Sample rate of a raw PCM file (default: 16000)")
    parser.add_argument("-l", "--language", help="Language hint, e.g. de, en, zh-CN, zh-TW (sent as ?language=...)")
    parser.add_argument("-t", "--token", default=os.getenv("API_TOKEN"), help="API token (default: $API_TOKEN)")
    args = parser.parse_args()

    endpoint = args.endpoint
    if args.language:
        endpoint += ("&" if "?" in endpoint else "?") + urlencode({"language": args.language})

    print(f"Connecting to {endpoint}...")

    pcm, rate = load_audio(args.file, args.rate)
    duration = len(pcm) / (2.0 * rate)
    print(f"Audio Duration: {duration:.2f}s")

    start_time = time.time()
    try:
        headers = {"Authorization": f"Bearer {args.token}"} if args.token else None
        async with websockets.connect(endpoint, max_size=None, additional_headers=headers) as ws:
            await asyncio.gather(sender(ws, pcm, rate, args.file), receiver(ws))

        end_time = time.time()
        process_time = end_time - start_time
        rtf = process_time / duration
        print(f"\nProcessing Time: {process_time:.2f}s")
        print(f"Real-Time Factor (RTF): {rtf:.4f}")

    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
