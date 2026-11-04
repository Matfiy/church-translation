from fastapi.responses import HTMLResponse 
import sounddevice as sd    
import numpy as np      
from TTS.api import TTS     
import mlx_whisper  
import requests     
from fastapi import FastAPI, WebSocket, File, UploadFile
import uvicorn      
import asyncio
import threading
import queue
import collections
import concurrent.futures
import base64  
import os
import re
import time
import lameenc
from dotenv import load_dotenv

# API keys live in .env (not committed to git). See .env.example.
load_dotenv()
FISH_AUDIO_API_KEY = os.getenv("FISH_AUDIO_API_KEY")
if not FISH_AUDIO_API_KEY:
    print("WARNING: FISH_AUDIO_API_KEY is not set. Copy .env.example to .env and add your key.")
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")
if not GOOGLE_TTS_API_KEY:
    print("WARNING: GOOGLE_TTS_API_KEY is not set. Ukrainian audio will not play until you add it to .env.")
# Gemini voices need a separate auth key bound to a service account (Agent Platform API)
GOOGLE_GEMINI_TTS_KEY = os.getenv("GOOGLE_GEMINI_TTS_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
if not GOOGLE_GEMINI_TTS_KEY or not GOOGLE_CLOUD_PROJECT:
    print("WARNING: GOOGLE_GEMINI_TTS_KEY or GOOGLE_CLOUD_PROJECT is not set. Gemini Ukrainian voices will not play.")
GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
GEMINI_TTS_STYLE = "Read this calmly and warmly, like a pastor preaching a sermon"

audio_queue = queue.Queue()

app = FastAPI()

# Runs translate+TTS+broadcast for each phrase off the main capture thread,
# one at a time (max_workers=1 keeps phrases broadcast in the order spoken)
# so the mic keeps recording the next phrase instead of waiting on this one.
translation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="translator")

# separate guest lists for eng and ukr
clients = {
    "uk": set(),
    "en": set()
}

admin_clients = set()

# (language, text) of the last few translated phrases, given to the translator as context
recent_sentences = collections.deque(maxlen=3)

current_source_lang = "uk"

# Voice options, keyed by broadcast language. Ukrainian ids are Google Cloud
# TTS voice names ("gemini:" prefix = Gemini TTS voice); English ids are Fish
# Audio voice IDs. Add more {"name": ..., "id": ...} entries here as you get
# more voices. The first entry is the default at startup.
VOICE_OPTIONS = {
    "uk": [
        {"name": "Gemini Charon (male)", "id": "gemini:Charon"},
        {"name": "Gemini Kore (female)", "id": "gemini:Kore"},
        {"name": "Google WaveNet (female)", "id": "uk-UA-Wavenet-B"},
        {"name": "Google Chirp HD Charon (male)", "id": "uk-UA-Chirp3-HD-Charon"},
        {"name": "Google Chirp HD Kore (female)", "id": "uk-UA-Chirp3-HD-Kore"},
        {"name": "Google Standard (female)", "id": "uk-UA-Standard-B"},
    ],
    "en": [
        {"name": "Adrian", "id": "bf322df2096a46f18c579d0baa36f41d"},
        {"name": "Default English", "id": "da2053129ddc47e5b97fa33b1ddbcef9"},
    ],
}

current_voice_id = {
    "uk": VOICE_OPTIONS["uk"][0]["id"],
    "en": VOICE_OPTIONS["en"][0]["id"],
}

# master flag for controling audio
is_capturing = False

# Comma-separated glossary of proper nouns, scripture references, and unusual
# vocabulary extracted from uploaded sermon notes. Fed into both Whisper's
# initial_prompt and the translation prompt so names/terms stay consistent.
sermon_glossary = ""

# === config ===
SAMPLE_RATE = 48000
CHUNK_DURATION = 3

# Which Dante input channels to capture, 1-indexed as labeled in Dante Controller
# (e.g. [1, 2] for channels 1 & 2, [5] for just channel 5, [3, 4] for 3 & 4).
DANTE_CHANNELS = [63, 64]

# function to locate the dante device id
def get_dante_id():
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        name = dev['name'].lower()
        if "dante" in name and dev['max_input_channels'] > 0:
            print(f"Success! Found Dante stream at ID #{idx} ({dev['name']})")
            return idx
            
    print("\nWARNING: Auto-search failed. Forcing connection to Device ID 1.")
    return 1 # Hardcoded to your Dante Virtual Soundcard
            
    print("\nCRITICAL WARNING: Could not find PythonDante!")
    return 0

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
    if language not in clients:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    clients[language].add(websocket) 
    print(f"Client joined {language.upper()}. Total listeners: {len(clients[language])}")
    await broadcast_listener_counts()

    try:
        while True:
            await websocket.receive_text() 
    except Exception:
        pass
    finally:
        clients[language].discard(websocket)
        print(f"Client disconnected from {language.upper()}. Total listeners: {len(clients[language])}")
        await broadcast_listener_counts()

# Serve the Admin Dashboard Page
@app.get("/admin", response_class=HTMLResponse)
async def get_admin_dashboard():
    with open("admin.html", "r", encoding="utf-8") as f:
        return f.read()

# WebSocket specifically for the Tech Booth Admin Dashboard
@app.websocket("/ws/admin")
async def admin_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    admin_clients.add(websocket)
    await websocket.send_json(listener_counts())
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        admin_clients.discard(websocket)

# logic for start/stop button
@app.post("/admin/set_capture/{state}")
async def set_capture_state(state: bool):
    global is_capturing
    is_capturing = state
    status_str = "STARTED" if is_capturing else "PAUSED"
    print(f"\n[ADMIN] Sermon Audio Capture: {status_str}")

    # empty out queue when paused so worship audio doesn't stack up
    if not is_capturing:
        with audio_queue.mutex:
            audio_queue.queue.clear()

    return {"status": "success", "is_capturing": is_capturing}

# Accepts uploaded sermon notes (plain text), extracts a glossary of proper
# nouns/scripture references/unusual vocabulary via the local LLM, and stores
# it for use as transcription/translation context during the service.
@app.post("/admin/upload_notes")
async def upload_notes(file: UploadFile = File(...)):
    global sermon_glossary

    raw_bytes = await file.read()
    try:
        notes_text = raw_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return {"status": "error", "message": "Could not read file as text. Please upload a plain .txt file."}

    if not notes_text:
        return {"status": "error", "message": "Uploaded file is empty."}

    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "llama3:8b",
            "system": "You extract a short glossary from sermon notes for a live speech transcription and translation system. List only proper nouns, names, unusual vocabulary, and scripture references that a transcription engine might mishear or a translator might render inconsistently. Output ONLY a comma-separated list, nothing else, no more than 40 items.",
            "prompt": notes_text[:6000],
            "stream": False
        })
        sermon_glossary = response.json()['response'].strip()
    except Exception as e:
        return {"status": "error", "message": f"Failed to extract glossary: {e}"}

    print(f"\n[ADMIN] Sermon notes uploaded. Extracted glossary: {sermon_glossary}")
    return {"status": "success", "glossary": sermon_glossary}

# Clears the stored glossary, e.g. between services
@app.post("/admin/clear_notes")
async def clear_notes():
    global sermon_glossary
    sermon_glossary = ""
    print("\n[ADMIN] Sermon notes glossary cleared")
    return {"status": "success"}

# Lets the admin dashboard show what glossary is currently active
@app.get("/admin/notes_status")
async def notes_status():
    return {"glossary": sermon_glossary}

# Lists available Fish Audio voices so the admin dashboard can populate its dropdowns
@app.get("/admin/voices")
async def get_voices():
    return {"options": VOICE_OPTIONS, "current": current_voice_id}

# Sets which Fish Audio voice ID to use for a given broadcast language
@app.post("/admin/set_voice/{lang}/{voice_id}")
async def set_voice(lang: str, voice_id: str):
    global current_voice_id
    if lang not in VOICE_OPTIONS:
        return {"status": "error"}
    if not any(v["id"] == voice_id for v in VOICE_OPTIONS[lang]):
        return {"status": "error"}
    current_voice_id[lang] = voice_id
    print(f"\n[ADMIN] {lang.upper()} voice set to: {voice_id}")
    return {"status": "success", "lang": lang, "voice_id": voice_id}


# Update broadcast_payload to also notify the admin dashboard
async def broadcast_payload(payload: dict, target_language: str):
    # Send to active listeners
    for client in list(clients[target_language]): 
        try:
            await client.send_json(payload) 
        except:
            pass 
            
    # Send to Tech Booth Admin Dashboard
    admin_data = {
        "type": "transcript",
        "original": payload["original"],
        "translated": payload["translated"],
        "lang": payload["lang"]
    }
    for admin in list(admin_clients):
        try:
            await admin.send_json(admin_data)
        except:
            pass

def listener_counts():
    return {"type": "listeners", "en": len(clients["en"]), "uk": len(clients["uk"])}

# Pushes the current listener numbers to every open admin dashboard
async def broadcast_listener_counts():
    counts = listener_counts()
    for admin in list(admin_clients):
        try:
            await admin.send_json(counts)
        except:
            pass

# NEW: The lightweight callback that just fills the bucket
def audio_callback(indata, frames, time, status):
    if status:
        pass
    # indata has one column per opened channel (1..N); pull out just the
    # Dante channels we actually want (0-indexed into the opened block).
    selected = indata[:, [ch - 1 for ch in DANTE_CHANNELS]]
    audio_queue.put(selected.copy())

# NEW: The background worker that waits for 3 seconds of audio, then translates
def transcription_worker():
    global is_capturing

    # Smart VAD Thresholds (one audio chunk = 4096 frames at 48kHz, about 85ms)
    SILENCE_THRESHOLD = 0.015
    MIN_CHUNKS = 12 # ~1.0 second minimum audio length
    MAX_CHUNKS = 211 # ~18 seconds maximum audio length (force cut)
    QUIET_SEARCH_CHUNKS = 35 # a force cut looks back ~3 seconds for the quietest moment
    PRE_ROLL_CHUNKS = 4 # ~0.35 seconds kept from just before speech starts
    STOPPED_TALKING_CHUNKS = 23 # ~2 seconds of silence sends a held unfinished sentence anyway
    FLUSH_FRAGMENT_CHUNKS = 176 # ~15 seconds with no sentence ending: send it rather than wait longer
    CARRY_PADDING_SECONDS = 0.1 # audio kept before a held sentence's first word

    SENTENCE_ENDINGS = (".", "!", "?", "…")
    CLOSING_CHARS = "\"'”’)]»" # trailing quotes/brackets, e.g. closing a scripture quote
    HALLUCINATIONS = ["thank you", "thanks for watching", "i'll call you my friend", "so it's a little later", "дякую", "you", "Let us open in prayer.", "Let us pray.", "the word of God", "word of god",
                      "Welcome, everyone.", "Today we will look closely at the Word of God.",
                      "Ласкаво просимо.", "Сьогодні ми уважно розглянемо Слово Боже.", "Слово Боже", "Помолімося разом.",
                      # YouTube sign-offs Whisper picked up in training and invents during quiet moments
                      "Subscribe to the channel", "Subscribe to my channel", "Subscribe to our channel", "Please subscribe", "Like and subscribe",
                      "Don't forget to subscribe", "Thank you for watching", "See you in the next video",
                      "Підписуйтесь на канал", "Підпишіться на канал", "Підписуйтесь на наш канал", "Дякую за перегляд", "Дякуємо за перегляд", "Дякую за увагу", "До зустрічі в наступному відео"]

    # Whisper mimics the punctuation/formatting style of whatever text it's
    # given as context, so priming it with a fully-punctuated example nudges
    # it to keep ending sentences with "." "!" "?" instead of dropping them.
    # Keep these free of sermon-sounding phrases: Whisper echoes its prompt
    # back as a hallucination during pauses (that's where "the Word of God" came from).
    initial_prompts = {
        "en": "Okay, so. Now, as I was saying, let's go on.",
        "uk": "Отже, так. Тепер, як я вже казав, продовжимо.",
    }

    def normalize(text):
        return re.sub(r"[^\w\s']", "", text.lower()).strip()

    # Hallucinated sentences are matched whole (not as substrings) so real
    # speech like "your" or "we read the word of God daily" isn't thrown away.
    # Prompt sentences are included too, since Whisper can echo them verbatim.
    hallucination_set = {normalize(h) for h in HALLUCINATIONS}
    for prompt in initial_prompts.values():
        hallucination_set.update(normalize(s) for s in re.split(r"(?<=[.!?…])\s+", prompt))
    hallucination_set.discard("")

    def ends_sentence(text):
        return text.rstrip(CLOSING_CHARS).endswith(SENTENCE_ENDINGS)

    # Splits the transcription into sentences with the time each one starts,
    # dropping hallucinated sentences from anywhere (start, middle, or end)
    def split_sentences(result):
        sentences = []
        def add(text, start):
            text = text.strip()
            cleaned = normalize(text)
            if not cleaned:
                return
            if cleaned in hallucination_set or cleaned.startswith("todays terms"):
                print(f"[Dropped hallucination] {text}")
                return
            sentences.append({"text": text, "start": start})

        words = []
        start = None
        for segment in result.get("segments", []):
            for word in segment.get("words") or []:
                if start is None:
                    start = word["start"]
                words.append(word["word"])
                if ends_sentence(word["word"].strip()):
                    add("".join(words), start)
                    words = []
                    start = None
        if words:
            add("".join(words), start)
        return sentences

    def chunk_volume(data):
        if data.shape[1] > 1:
            return np.max(np.abs(np.mean(data, axis=1)))
        return np.max(np.abs(data.flatten()))

    # A longer phrase takes a shorter pause to end. Don't go below ~0.5 seconds:
    # Whisper puts a false period at a mid-sentence breath if the audio stops there.
    def pause_chunks_needed(total_chunks):
        if total_chunks < 47: # under ~4 seconds
            return 8 # ~0.7 second pause
        return 6 # ~0.5 second pause

    # Audio of a sentence that wasn't finished yet. It goes back through Whisper
    # together with what comes next, so the whole sentence is heard at once.
    carried_chunks = []

    while True:
        # if capture paused, clear queue and sleeo
        if not is_capturing:
            with audio_queue.mutex:
                audio_queue.queue.clear()
            carried_chunks = []
            recent_sentences.clear()
            time.sleep(0.5)
            continue

        accumulated_audio = carried_chunks
        volumes = [chunk_volume(chunk) for chunk in accumulated_audio]
        has_carried_audio = bool(carried_chunks)
        carried_chunks = []
        new_speech = False
        silence_counter = 0
        cut_reason = "stopped"

        # dynamic gathering loop
        while is_capturing:
            try:
                data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # check volume of tiny audio slice
            volume = chunk_volume(data)
            accumulated_audio.append(data)
            volumes.append(volume)

            # If it's quiet, count up the silence. If he speaks, reset the counter!
            if volume < SILENCE_THRESHOLD:
                silence_counter += 1
            else:
                silence_counter = 0
                new_speech = True

            if not new_speech:
                if has_carried_audio:
                    # he finished talking mid-sentence: stop waiting for the rest
                    if silence_counter >= STOPPED_TALKING_CHUNKS:
                        cut_reason = "stopped"
                        break
                else:
                    # nothing said yet: keep a short lead-in, not the silence
                    accumulated_audio = accumulated_audio[-PRE_ROLL_CHUNKS:]
                    volumes = volumes[-PRE_ROLL_CHUNKS:]
                continue

            total_chunks = len(accumulated_audio)

            # trigger 1: preacher took a breath (silence) and spoke long enough
            if silence_counter >= pause_chunks_needed(total_chunks) and total_chunks >= MIN_CHUNKS:
                cut_reason = "pause"
                break

            # trigger 2: preacher has not taken a breath in 18 seconds (force cut
            # at the quietest recent moment, keeping the rest for the next phrase)
            if total_chunks >= MAX_CHUNKS:
                search_from = total_chunks - QUIET_SEARCH_CHUNKS
                cut_at = search_from + int(np.argmin(volumes[search_from:])) + 1
                carried_chunks = accumulated_audio[cut_at:]
                accumulated_audio = accumulated_audio[:cut_at]
                cut_reason = "max length"
                break

        # paused before anything was said
        if not new_speech and not has_carried_audio:
            continue

        # 2. Stitch chunks together
        indata = np.concatenate(accumulated_audio, axis=0)

        # 3. Mix stereo to mono
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1).astype(np.float32)
        else:
            audio_data = indata.flatten().astype(np.float32)

        # 4. The Sanity Check
        max_volume = np.max(np.abs(audio_data))
        print(f"Volume: {max_volume:.4f}")

        # If it's pure silence, skip it and wait for the next bucket
        if max_volume < 0.001:
            continue

        # downsample from 48kHz to 16kHz for Whisper
        audio_16k = audio_data[::3]

        mlx_model_repo = "mlx-community/whisper-large-v3-mlx"

        # Prime Whisper with today's proper nouns/scripture references so it
        # doesn't have to guess unfamiliar words purely from audio
        whisper_prompt = initial_prompts.get(current_source_lang)
        if sermon_glossary and whisper_prompt:
            whisper_prompt = f"{whisper_prompt} Today's terms: {sermon_glossary}"

        try:
            if current_source_lang == "auto":
                result = mlx_whisper.transcribe(
                    audio_16k,
                    path_or_hf_repo=mlx_model_repo,
                    word_timestamps=True,
                    hallucination_silence_threshold=1.0
                )
            else:
                # word_timestamps + hallucination_silence_threshold let Whisper
                # skip text it invents inside silent gaps (>1s) mid-chunk,
                # which is where mid-phrase hallucinations come from
                result = mlx_whisper.transcribe(
                    audio_16k,
                    path_or_hf_repo=mlx_model_repo,
                    language=current_source_lang,
                    initial_prompt=whisper_prompt,
                    word_timestamps=True,
                    hallucination_silence_threshold=1.0
                )

            detected_lang = result.get("language", "unknown")
            sentences = split_sentences(result)

        except Exception as e:
            print(f"MLX Transcription Error: {e}")
            continue

        # A chunk is "noise" if it's the wrong language or nothing is left
        # after split_sentences removed the known Whisper hallucinations
        if detected_lang not in ["en", "uk"] or not sentences:
            continue

        # Whisper tends to add a period wherever the audio stops, so after a
        # force cut the last sentence can't be trusted even if it has one
        last_is_finished = ends_sentence(sentences[-1]["text"]) and cut_reason != "max length"
        complete = sentences if last_is_finished else sentences[:-1]
        is_complete = True

        if not last_is_finished:
            if cut_reason == "stopped" or (not complete and len(accumulated_audio) >= FLUSH_FRAGMENT_CHUNKS):
                # nothing more is coming, or it has run too long without an ending
                complete = sentences
                is_complete = False
            else:
                # Hold the unfinished sentence: its audio goes back into the next phrase
                tail_start = max(0, int((sentences[-1]["start"] - CARRY_PADDING_SECONDS) * SAMPLE_RATE))
                tail = indata[tail_start:]
                tail_chunks = [tail[i:i + 4096] for i in range(0, len(tail), 4096)]
                # no need to keep the silence after it, just a short gap
                while len(tail_chunks) > 2 and chunk_volume(tail_chunks[-1]) < SILENCE_THRESHOLD and chunk_volume(tail_chunks[-2]) < SILENCE_THRESHOLD:
                    tail_chunks.pop()
                carried_chunks = tail_chunks + carried_chunks
                print(f"\n[Holding unfinished sentence, {cut_reason}] {sentences[-1]['text']}")

        if not complete:
            continue

        spoken_text = " ".join(sentence["text"] for sentence in complete)
        print(f"\n[Detected: {detected_lang.upper()}, {cut_reason}] {spoken_text}")

        # Nudge the translation model to keep sermon-specific terms consistent
        glossary_line = f"Glossary of today's sermon terms (keep these spellings/names consistent): {sermon_glossary}. " if sermon_glossary else ""

        # Hand this phrase off to the translator thread and immediately go
        # back to listening for the next one instead of blocking on it here
        translation_executor.submit(translate_and_broadcast, spoken_text, detected_lang, glossary_line, is_complete)


# Llama sometimes opens with "Here is the direct English translation:" even when told not to
TRANSLATION_PREAMBLE = re.compile(
    r"""^\s*(?:
        (?:here\s+is|here's|here’s)\b[^:\n]{0,60}\btranslation\b[^:\n]{0,30}:
        | (?:the\s+)?(?:direct\s+)?(?:english\s+|ukrainian\s+)?translation\s*:
        | ось\b[^:\n]{0,60}\bпереклад\w*[^:\n]{0,30}:
        | переклад\s*:
    )\s*""",
    re.IGNORECASE | re.VERBOSE,
)

def strip_preamble(text):
    text = TRANSLATION_PREAMBLE.sub("", text).strip()
    # the preamble usually comes with the translation wrapped in quotes
    if len(text) > 1 and text[0] in "\"“«" and text[-1] in "\"”»":
        text = text[1:-1].strip()
    return text


# TTS needs ending punctuation to sound right. A sentence that got cut short
# trails off with "…" instead of sounding like it finished.
def finish_punctuation(text, is_complete):
    if not is_complete:
        return text.rstrip(" .,!?…") + "…"
    return text if text.endswith((".", "!", "?", "…")) else text + "."


# Translates one already-transcribed phrase, synthesizes TTS audio, and
# broadcasts it. Runs on translation_executor so it never blocks the mic
# capture loop in transcription_worker.
def translate_and_broadcast(spoken_text, detected_lang, glossary_line, is_complete=True):
    thread_id = threading.get_ident()
    temp_filename = f"output_{thread_id}.wav"

    # The last few sentences, so the translator knows who "he" or "this" refers to
    previous = " ".join(text for lang, text in recent_sentences if lang == detected_lang)
    context_line = f'For context only, do not translate this part. The preacher just said: "{previous}"\n' if previous else ""
    recent_sentences.append((detected_lang, spoken_text))

    try:
            # PATH A: Pastor speaks English -> Translate to Ukrainian
            if detected_lang == "en" and len(clients["uk"]) > 0:
                response = requests.post('http://localhost:11434/api/generate', json={
                    "model": "llama3:8b",
                    "system": "You are a raw translation machine. You must output ONLY the direct translation. Never include notes, explanations, introductions, or conversational filler.",
                    "prompt": f"{glossary_line}{context_line}Translate this English church sermon phrase to natural Ukrainian: {spoken_text}",
                    "stream": False
                })
                translated_text = response.json()['response'].strip()
                
                # The Python Guillotine
                translated_text = translated_text.split("Note:")[0].split("Notes:")[0].split("Примітка:")[0].strip()
                translated_text = strip_preamble(translated_text)
                if not translated_text:
                    return
                
                # The Anti-Moan Punctuation Lock
                translated_text = finish_punctuation(translated_text, is_complete)
                
                uk_voice = current_voice_id["uk"]
                if uk_voice.startswith("gemini:"):
                    # Gemini TTS API Call for Ukrainian (via Agent Platform)
                    google_response = requests.post(
                        f"https://aiplatform.googleapis.com/v1/projects/{GOOGLE_CLOUD_PROJECT}/locations/global/publishers/google/models/{GEMINI_TTS_MODEL}:generateContent?key={GOOGLE_GEMINI_TTS_KEY}",
                        json={
                            "contents": [{"role": "user", "parts": [{"text": f"{GEMINI_TTS_STYLE}: {translated_text}"}]}],
                            "generationConfig": {
                                "responseModalities": ["AUDIO"],
                                "speechConfig": {
                                    "languageCode": "uk-UA",
                                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": uk_voice.split(":", 1)[1]}}
                                }
                            }
                        }
                    )
                else:
                    # Google Cloud TTS API Call for Ukrainian
                    google_response = requests.post(
                        f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}",
                        json={
                            "input": {"text": translated_text},
                            "voice": {"languageCode": "uk-UA", "name": uk_voice},
                            "audioConfig": {"audioEncoding": "MP3"} # compressed to save upload bandwidth
                        }
                    )

                if google_response.status_code == 200:
                    if uk_voice.startswith("gemini:"):
                        # Gemini returns raw 24kHz 16-bit mono PCM, so compress it to MP3 to save upload bandwidth
                        pcm = base64.b64decode(google_response.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
                        encoder = lameenc.Encoder()
                        encoder.set_bit_rate(48)
                        encoder.set_in_sample_rate(24000)
                        encoder.set_channels(1)
                        encoder.set_quality(2)
                        mp3_bytes = encoder.encode(pcm) + encoder.flush()
                        audio_base64 = base64.b64encode(mp3_bytes).decode('utf-8')
                    else:
                        # Google already returns the audio as base64, so it goes straight into the payload
                        audio_base64 = google_response.json()["audioContent"]

                    payload = {"original": spoken_text, "translated": translated_text, "audio": audio_base64, "lang": "Ukrainian"}
                    asyncio.run_coroutine_threadsafe(broadcast_payload(payload, "uk"), server_loop)
                else:
                    print(f"Google TTS Error (UK): {google_response.text}")

            # PATH B: Pastor speaks Ukrainian -> Translate to English
            elif detected_lang == "uk" and len(clients["en"]) > 0:
                response = requests.post('http://localhost:11434/api/generate', json={
                    "model": "llama3:8b",
                    "system": "You are a raw translation machine. You must output ONLY the direct English translation. Never include notes, explanations, introductions, or quotes.",
                    "prompt": f"{glossary_line}{context_line}Translate this Ukrainian church sermon phrase to natural English: {spoken_text}",
                    "stream": False
                })
                translated_text = response.json()['response'].strip()
                
                # The Python Guillotine
                translated_text = translated_text.split("Note:")[0].split("Notes:")[0].strip()
                translated_text = strip_preamble(translated_text)
                translated_text = translated_text.replace('"', '').replace("'", "").replace("*", "").strip()
                
                # The Anti-Moan Punctuation Lock
                translated_text = finish_punctuation(translated_text, is_complete)
                
                if len(translated_text.split()) < 3:
                    print(f"Skipping English TTS for short phrase: {translated_text}")
                    return
                
                # NEW: Fish Audio API Call
                fish_response = requests.post(
                    "https://api.fish.audio/v1/tts",
                    headers={
                        "Authorization": f"Bearer {FISH_AUDIO_API_KEY}",
                        "Content-Type": "application/json",
                        "model": "s2.1-pro-free" # Use "s2.1-pro" for the paid production tier
                    },
                    json={
                        "text": translated_text,
                        "reference_id": current_voice_id["en"],
                        "format": "mp3", # compressed to save upload bandwidth
                        "mp3_bitrate": 64
                    }
                )
                
                if fish_response.status_code == 200:
                    # Fish API returns raw bytes, so we encode it directly to base64 and bypass the hard drive
                    audio_base64 = base64.b64encode(fish_response.content).decode('utf-8')
                    
                    payload = {"original": spoken_text, "translated": translated_text, "audio": audio_base64, "lang": "English"}
                    asyncio.run_coroutine_threadsafe(broadcast_payload(payload, "en"), server_loop)
                else:
                    print(f"Fish Audio Error: {fish_response.text}")

    except Exception as e:
            print(f"Pipeline Error: {e}")
    finally:
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass

@app.on_event("startup")
async def startup_event():
    global audio_stream, server_loop
    server_loop = asyncio.get_running_loop() 

    # NEW: Start the background worker thread
    threading.Thread(target=transcription_worker, daemon=True).start()

    print("Connecting to Dante network...")
    try:
        audio_stream = sd.InputStream(
            device=DANTE_INPUT_ID,
            channels=max(DANTE_CHANNELS), # open enough channels to reach the highest one selected
            samplerate=SAMPLE_RATE,
            blocksize=4096, # NEW: The tiny 4096-frame bucket size!
            callback=audio_callback
        )
        audio_stream.start() 
        print("Dante stream successfully started!")
    except Exception as e:
        print(f"Failed to initialize Dante input device #{DANTE_INPUT_ID}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)