# Client for streaming ASR - this is for TESTING PURPOSES

import asyncio
import json
import argparse
import sys
import websockets
from datetime import datetime
from urllib.parse import urlencode

import os
import time

CHUNK_BYTES = 640  # 20ms at 16kHz 16-bit mono

async def sender(ws, pcm_path: str):
    # Handshake / Config
    await ws.send(json.dumps({
        "type": "start",
        "format": "pcm_s16le",
        "sample_rate_hz": 16000,
        "channels": 1
    }))

    print(f"Streaming {pcm_path}...")
    with open(pcm_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            await ws.send(chunk)
            await asyncio.sleep(0) # Yield control to ensure receiver can process messages

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
    parser.add_argument("-f", "--file", required=True, help="Path to raw PCM 16k 16-bit mono file (or WAV with correct format)")
    parser.add_argument("-l", "--language", help="Language hint, e.g. de, en, zh-CN, zh-TW (sent as ?language=...)")
    parser.add_argument("-t", "--token", default=os.getenv("API_TOKEN"), help="API token (default: $API_TOKEN)")
    args = parser.parse_args()

    endpoint = args.endpoint
    if args.language:
        endpoint += ("&" if "?" in endpoint else "?") + urlencode({"language": args.language})

    print(f"Connecting to {endpoint}...")
    
    file_size = os.path.getsize(args.file)
    duration = file_size / 32000.0 # 16000 * 2 bytes
    print(f"Audio Duration: {duration:.2f}s")
    
    start_time = time.time()
    try:
        headers = {"Authorization": f"Bearer {args.token}"} if args.token else None
        async with websockets.connect(endpoint, max_size=None, additional_headers=headers) as ws:
            await asyncio.gather(sender(ws, args.file), receiver(ws))
            
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
