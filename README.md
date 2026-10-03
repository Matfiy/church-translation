# church-translation

Live sermon translation for FUEBC. It listens to the pastor's microphone over Dante, transcribes and translates each phrase between Ukrainian and English, and speaks the translation to listeners' phones in their chosen language.

Listeners open a web page on the church Wi-Fi, pick a language, and hear the translated audio with the text on screen. A tech-booth admin page controls the source language, starts and stops the feed, switches voices, and shows a live transcript.

## How it works

```
Dante mic ──► Whisper large-v3 (MLX) ──► Llama 3 8B (Ollama) ──► Fish Audio TTS ──► listeners' browsers
              speech → text               translate uk ⇄ en       text → speech       (WebSocket)
```

- **Audio**: captured from the Dante Virtual Soundcard and split into phrases at the pastor's pauses (a 12-second phrase is cut off even without a pause).
- **Transcription**: runs locally on the Mac with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). Common Whisper hallucinations (e.g. "Thank you.", "The Word of God.") are filtered out sentence by sentence.
- **Translation**: runs locally with Llama 3 8B through [Ollama](https://ollama.com). A phrase is only translated when someone is listening in the target language.
- **Voice**: synthesized by the [Fish Audio](https://fish.audio) API (needs an API key and internet).

## Requirements

- A Mac with Apple Silicon (M1 or later). mlx-whisper does not run on Intel Macs or other platforms.
- Python 3.10
- [Ollama](https://ollama.com) with the `llama3:8b` model
- [Dante Virtual Soundcard](https://www.getdante.com/products/software-essentials/dante-virtual-soundcard/) receiving the pastor's mic
- A Fish Audio API key

## Setup

```bash
git clone https://github.com/Matfiy/church-translation.git
cd church-translation

# Python environment
python3.10 -m venv ai_env
source ai_env/bin/activate
pip install fastapi uvicorn python-multipart mlx-whisper sounddevice numpy requests python-dotenv coqui-tts "scipy==1.14.1"

# Translation model
ollama pull llama3:8b

# API key
cp .env.example .env
# then edit .env and set FISH_AUDIO_API_KEY
```

> `scipy` is pinned to 1.14.1 because the scipy 1.15.x wheels fail to load on recent macOS with `ImportError: ... section '__DATA/__thread_bss' has a zero-fill section type`.

The first run downloads the Whisper model (`mlx-community/whisper-large-v3-mlx`, about 3 GB).

## Running a service

1. Make sure Ollama is running (`ollama serve`, or open the Ollama app).
2. Start the server:
   ```bash
   source ai_env/bin/activate
   python server.py
   ```
   It listens on port 8000 on all network interfaces.
3. In the tech booth, open the admin console at `http://<mac-ip>:8000/admin`:
   - set the **source language** (Ukrainian, English, or auto-detect; defaults to Ukrainian)
   - optionally upload **sermon notes** as a `.txt` file — names and scripture references are extracted into a glossary so Whisper and the translator spell them consistently
   - press **Start** on the translation feed (capture is off when the server starts)
4. Listeners open `http://<mac-ip>:8000/` on their phones and choose a language.

Find the Mac's IP address in System Settings → Wi-Fi → Details, or with `ipconfig getifaddr en0`.

## Configuration

Settings are constants near the top of `server.py`:

| Setting | Default | What it does |
|---|---|---|
| `DANTE_CHANNELS` | `[63, 64]` | Dante input channels to capture, numbered as in Dante Controller |
| `current_source_lang` | `"uk"` | Starting source language (`"en"`, `"uk"`, or `"auto"`); also set from the admin page |
| `VOICE_OPTIONS` | 1 Ukrainian, 2 English | Fish Audio voice IDs shown in the admin voice picker. Add `{"name": ..., "id": ...}` entries for more |

The pause detection thresholds (`SILENCE_THRESHOLD`, `PAUSE_CHUNKS`, `MAX_CHUNKS`) and the hallucination list (`HALLUCINATIONS`) are at the top of `transcription_worker()`.

Secrets go in `.env`, which git ignores:

| Variable | Required | Description |
|---|---|---|
| `FISH_AUDIO_API_KEY` | yes | Fish Audio API key used for all speech synthesis |

## Troubleshooting

**No audio / "Failed to initialize Dante input device".** The server looks for an input device with "dante" in its name and falls back to device ID 1. Two helper scripts help find the right one:

```bash
python find_audio.py   # list all audio devices and their IDs
python xray.py         # show which channels on device 1 currently have signal
```

Then update `DANTE_CHANNELS` (and the fallback ID in `get_dante_id()` if needed).

**Translations never appear.** Check that Ollama is running and has the model: `ollama list` should show `llama3:8b`. Also check that the feed is started on the admin page and that at least one listener is connected in the target language.

**Text appears but no audio.** Look for `Fish Audio Error` in the server output — usually a missing or invalid `FISH_AUDIO_API_KEY`. English translations under 3 words are intentionally dropped (neither shown nor spoken).

**A phrase keeps showing up that the pastor didn't say.** Add it to `HALLUCINATIONS` in `server.py`. Matching is per whole sentence, so real speech that merely contains the phrase is kept.

## Project layout

```
server.py       FastAPI server, audio capture, transcription, translation, broadcasting
index.html      listener page (/)
admin.html      tech-booth console (/admin)
find_audio.py   lists audio devices
xray.py         scans Dante channels for signal
.env.example    template for .env
```

## License

No license has been chosen yet. Until one is added, all rights are reserved by the authors.
