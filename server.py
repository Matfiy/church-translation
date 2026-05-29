import sounddevice as sd    # connects python to computer's audio hardware
import numpy as np      # industry standard for heavy-duty mathematics
from faster_whisper import WhisperModel # OpenAI's whisper model
import requests     # simple HTTP library used to send web requests out to the internet
from fastapi import FastAPI, WebSocket  # web framework for building API's in Python
import uvicorn      # web server production engine that executes the FastAPI code
import asyncio      # enables asynchronus programming (multiple tasks at the same time)
from TTS.api import TTS     # Coqui's advanced Text-to-Speech framework
import threading    # allows script to run completely seperate execution threads

