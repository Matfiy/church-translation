from fastapi.responses import HTMLResponse 
import sounddevice as sd    
import numpy as np      
from TTS.api import TTS     
from faster_whisper import WhisperModel 
import requests     
from fastapi import FastAPI, WebSocket  
import uvicorn      
import asyncio      
import threading    
import base64  # NEW: Used to encode audio into a text-safe format
import os      # NEW: Used to clean up temporary audio files

app = FastAPI()
clients = set()

print("Loading Whisper Model into VRAM...")
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

print("Loading TTS Model into VRAM...")
tts = TTS("tts_models/uk/mai/vits").to("cuda")

# === config ===
DANTE_INPUT_ID = 90
SAMPLE_RATE = 48000
CHUNK_DURATION = 3 

audio_stream = None
server_loop = None 

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.websocket("/stream/uk") 
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket) 
    print(f"Client connected. Total listeners: {len(clients)}")

    try:
        while True:
            await websocket.receive_text() 
    except Exception:
        pass
    finally:
        clients.remove(websocket) 
        print(f"Client disconnected. Total listeners: {len(clients)}")

# FIXED: Broadcasts a unified JSON package containing both text and audio
async def broadcast_payload(payload: dict):
    for client in clients: 
        try:
            await client.send_json(payload) 
        except:
            pass 

def process_audio(indata):
    # format the audio for Whisper
    audio_data = indata.flatten().astype(np.float32)

    # downsample from 48kHz to 16kHz for Whisper
    audio_16k = audio_data[::3]

    # transcribe to english
    segments, _ = whisper_model.transcribe(audio_16k, language="en")
    english_text = "".join([segment.text for segment in segments]).strip()

    # Block empty text and common Whisper hallucinations
    hallucinations = ["thank you", "thanks for watching", "i'll call you my friend", "so it's a little later"]
    
    # This now checks if ANY of the hallucination phrases are hidden inside the english text
    if not english_text or any(h in english_text.lower() for h in hallucinations):
        return
    
    print(f"\n[English] {english_text}")

    # translate to ukrainian via local Ollama
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "llama3:8b",
            "prompt": f"Translate this English church sermon phrase to natural Ukrainian. Only return the Ukrainian text, no quotes or explanations: {english_text}",
            "stream": False
        })
        ukrainian_text = response.json()['response'].strip()
        print(f"[Ukrainian] {ukrainian_text}")
    except Exception as e:
        print(f"Ollama Error: {e}")
        return
    
    # FIXED: Generate audio using a unique filename for this specific thread
    thread_id = threading.get_ident()
    temp_filename = f"output_{thread_id}.wav"
    
    try:
        tts.tts_to_file(
            text=ukrainian_text,
            file_path=temp_filename
        )

        # Read the audio data and convert it to a Base64 string
        with open(temp_filename, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Package everything together nicely
        payload = {
            "english": english_text,
            "ukrainian": ukrainian_text,
            "audio": audio_base64
        }

        # safely push the payload to the main server thread
        asyncio.run_coroutine_threadsafe(broadcast_payload(payload), server_loop)

    except Exception as e:
        print(f"TTS/Broadcast Error: {e}")
    finally:
        # Clean up the file from disk after sending
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass

def audio_callback(indata, frames, time, status):
    if status: 
        print(status) 
    threading.Thread(target=process_audio, args=(indata.copy(),)).start()

@app.on_event("startup")
async def startup_event():
    global audio_stream, server_loop
    server_loop = asyncio.get_running_loop() 

    print("Connecting to Dante network...")
    try:
        audio_stream = sd.InputStream(
            device=DANTE_INPUT_ID, 
            channels=1, 
            samplerate=SAMPLE_RATE,  
            blocksize=int(SAMPLE_RATE * CHUNK_DURATION), 
            callback=audio_callback 
        )
        audio_stream.start() 
        print("Dante stream successfully started!")
    except Exception as e:
        print(f"Failed to initialize Dante input device #{DANTE_INPUT_ID}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)