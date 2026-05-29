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

# seperate guest lists for eng and ukr
clients = {
    "uk": set(),
    "en": set()
}

current_source_lang = "auto"

print("Loading Whisper Model into VRAM...")
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

print("Loading Ukrainian TTS Model...")
tts_uk = TTS("tts_models/uk/mai/vits").to("cuda")

print("Loading English TTS Model...")
tts_en = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# === config ===
SAMPLE_RATE = 48000
CHUNK_DURATION = 3 

# function to locate the dante device id
def get_dante_id():
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if "DVS Recieve" in dev['name'] and "1-2" in dev['name'] and dev['max_input_channels'] > 0:
            return idx
    return 90

DANTE_INPUT_ID = get_dante_id()

audio_stream = None
server_loop = None 

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
    
# The Admin endpoint to lock the language
@app.post("/admin/set_language/{lang}")
async def set_admin_language(lang: str):
    global current_source_lang
    if lang in ["en", "uk", "auto"]:
        current_source_lang = lang
        print(f"\n[ADMIN] Source language firmly locked to: {lang.upper()}")
        return {"status": "success", "locked_to": lang}
    return {"status": "error"}

# dynamic websocket that accepts user's language choice
@app.websocket("/stream/{language}") 
async def websocket_endpoint(websocket: WebSocket, language: str):
    await websocket.accept()
    clients[language].add(websocket) 
    print(f"Client joined {language.upper()}. Total listeners: {len(clients[language])}")

    try:
        while True:
            await websocket.receive_text() 
    except Exception:
        pass
    finally:
        if language in clients:
            clients[language].remove(websocket) 
            print(f"Client disconnected from {language.upper()}. Total listeners: {len(clients[language])}")

# FIXED: Broadcasts a unified JSON package containing both text and audio
async def broadcast_payload(payload: dict, target_language: str):
    for client in clients[target_language]: 
        try:
            await client.send_json(payload) 
        except:
            pass 

def process_audio(indata):
    # format the audio for Whisper
    audio_data = indata.flatten().astype(np.float32)

    # downsample from 48kHz to 16kHz for Whisper
    audio_16k = audio_data[::3]

    if current_source_lang == "auto":
        segments, info = whisper_model.transcribe(audio_16k, vad_filter=True)
    else:
        # Force Whisper to strictly listen in the locked language
        segments, info = whisper_model.transcribe(audio_16k, vad_filter=True, language=current_source_lang)

    # DEFENSE 1: Turn on VAD (Voice Activity Detection)
    # This forces Whisper to completely ignore chunks of audio that don't contain actual human speech
    detected_lang = info.language
    spoken_text = "".join([segment.text for segment in segments]).strip()

    # DEFENSE 2: The Strict Language Lock
    # If Whisper somehow still guesses a random language, we silently kill the process right here
    if detected_lang not in ["en", "uk"]:
        return

    # Block empty text and common Whisper hallucinations
    hallucinations = ["thank you", "thanks for watching", "i'll call you my friend", "so it's a little later", "дякую"]
    if not spoken_text or any(h in spoken_text.lower() for h in hallucinations):
        return
    
    print(f"\n[Detected: {detected_lang.upper()}] {spoken_text}")

    thread_id = threading.get_ident()
    temp_filename = f"output_{thread_id}.wav"

    try:
        # PATH A: Pastor speaks English -> Translate to Ukrainian
        if detected_lang == "en" and len(clients["uk"]) > 0:
            response = requests.post('http://localhost:11434/api/generate', json={
                "model": "llama3:8b",
                "prompt": f"Translate this English church sermon phrase to natural Ukrainian. Only return the Ukrainian text: {spoken_text}",
                "stream": False
            })
            translated_text = response.json()['response'].strip()
            
            tts_uk.tts_to_file(text=translated_text, file_path=temp_filename)

            with open(temp_filename, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {"original": spoken_text, "translated": translated_text, "audio": audio_base64, "lang": "Ukrainian"}
            asyncio.run_coroutine_threadsafe(broadcast_payload(payload, "uk"), server_loop)

        # PATH B: Pastor speaks Ukrainian -> Translate to English
        elif detected_lang == "uk" and len(clients["en"]) > 0:
            response = requests.post('http://localhost:11434/api/generate', json={
                "model": "llama3:8b",
                "prompt": f"Translate this Ukrainian church sermon phrase to natural English. Do not include any introductory remarks, quotes, or conversational text. Only return the direct English translation: {spoken_text}",
                "stream": False
            })
            translated_text = response.json()['response'].strip()
            
            # Remove any stray markdown or quotes that make XTTS panic
            translated_text = translated_text.replace('"', '').replace("'", "").replace("*", "").strip()
            
            # FIXED: Avoid XTTS short-phrase melting. If it's less than 3 words, 
            # don't force a voice clone; drop it or use a threshold.
            if len(translated_text.split()) < 3:
                print(f"Skipping English TTS for short phrase to avoid distortion: {translated_text}")
                return
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            absolute_speaker_path = os.path.join(base_dir, "pastor_reference.wav")
            
            # FIXED: Added speed parameter (1.05) to prevent the audio from dragging or echoing
            tts_en.tts_to_file(
                text=translated_text, 
                speaker_wav=absolute_speaker_path, 
                language="en", 
                file_path=temp_filename,
                speed=1.05 
            )

            with open(temp_filename, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {"original": spoken_text, "translated": translated_text, "audio": audio_base64, "lang": "English"}
            asyncio.run_coroutine_threadsafe(broadcast_payload(payload, "en"), server_loop)

    except Exception as e:
        print(f"Pipeline Error: {e}")
    finally:
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