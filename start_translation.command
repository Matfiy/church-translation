#!/bin/zsh
# Starts the translation server. Add this file to System Settings → General →
# Login Items so it runs every time the Mac logs in. It opens in Terminal,
# which already has microphone access for the Dante input.

cd "$(dirname "$0")" || exit 1

# Ollama (translation model) — start the app if it isn't already running
open -ga Ollama
echo "Waiting for Ollama..."
for i in {1..60}; do
    curl -s http://localhost:11434 >/dev/null && break
    sleep 2
done

source ai_env/bin/activate

# Dante Virtual Soundcard can take a moment after login; without it the
# server would fall back to the wrong input device
echo "Waiting for the Dante input device..."
for i in {1..60}; do
    python -c "import sounddevice as sd, sys; sys.exit(0 if any('dante' in d['name'].lower() for d in sd.query_devices()) else 1)" 2>/dev/null && break
    sleep 2
done

echo "Starting translation server..."
python server.py
