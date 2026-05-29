import sounddevice as sd    # connects python to computer's audio hardware
import numpy as np      # industry standard for heavy-duty mathematics
from faster_whisper import WhisperModel # OpenAI's whisper model
import requests     # simple HTTP library used to send web requests out to the internet
from fastapi import FastAPI, WebSocket  # web framework for building API's in Python
import uvicorn      # web server production engine that executes the FastAPI code
import asyncio      # enables asynchronus programming (multiple tasks at the same time)
from TTS.api import TTS     # Coqui's advanced Text-to-Speech framework
import threading    # allows script to run completely seperate execution threads

# create the foundation for server
app = FastAPI()
clients = set()
loop = asyncio.get_event_loop()

# setup the translation engine
print("Loading Whisper Model into VRAM...")
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

print("Loading TTS Model into VRAM...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# === config ===
DANTE_INPUT_ID = 78 # Dante Recieve 1-2
SAMPLE_RATE = 48000
CHUNK_DURATION = 3 # process audio in 3 sec chunks

# setup for others to connect and listen
@app.websocket("/stream/uk") # opens a websocket at this URL
async def websocket_endpoint(webscoket: WebSocket):
    await websocket.accept()
    clients.add(websocket) # adds them to the clients guest list
    try:
        while True:
            await websocket.recieve_text() # an infinite pause button (to not close the websocket)
    except:
        clients.remove(websocket) # if the connection breaks, remove the client from the list

# function to send out the final translated audio to the clients
async def broadcast_audio(audio_bytes):
    for client in clients: # for everyone connected
        try:
            await client.send_bytes(audio_bytes) # send the translated audio file
        except:
            pass # if connection drops from a user, pass and dont crash for everyone else

# function to translate audio from eng -> ukr
def process_audio(indata):

    # format the audio for Whisper
    audio_data = indata.flatten().astype(np.float32)

    # transcribe to english
    segments, _ = whisper_model.transcribe(audio_data, language="en")
    english_text = "".join([segment.text for segment in segments]).strip()

    if not english_text:
        return # if there is 3 seconds of silence
    print(f"\n[English] {english_text}")

    # translate to ukrainian via local Ollama
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "llama3:8b",
            "prompt": f"Translate this English church sermon phrase to natural Ukrainian. Only return the Ukrainian text, no quotes or explanations: {english_text}",
            "stream": False
        })

        ukrainian_text = response.json()['response']
        print(f"[Ukrainian] {ukrainian_text}")
    except Exception as e:
        print(f"Ollama Error: {e}")
        return
    
    # make it voice
    try:
        tts.tts_to_file(
            text=ukrainian_text,
            speaker_wav="pastor_reference.wav",
            language="uk",
            file_path="output.wav"
        )

        # broadcast to all connected devices
        with open("output.wav", "rb") as f:
            audio_bytes = f.read()
        asyncio.run_coroutine_threadsafe(broadcast_audio(audio_bytes), loop)

    except Exception as e:
        print(f"TTS Error: {e}")

# make sure audio is being processed
def audio_callback(indata, frames, time, status):
    if status: # if any error
        print(status) # print the error

    # run in background
    threading.Thread(target=process_audio, args=(indata.copy(),)).start()