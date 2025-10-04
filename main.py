# requirements (example)
# pip install vosk elevenlabs fasttext-langdetect sounddevice scipy gpiozero

import os
import queue
import sounddevice as sd
import scipy.io.wavfile as wavfile
from gpiozero import Button
from vosk import Model, KaldiRecognizer
from elevenlabs import ElevenLabs, set_api_key, Voice  # adjust import per SDK version
from fasttext_langdetect import detect  # wrapper; name may vary

# ---------- config ----------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
set_api_key(ELEVENLABS_API_KEY)

SAMPLE_RATE = 16000
RECORD_SECONDS = 8  # how long to record (or implement VAD)
BUTTON_VICTIM_GPIO = 17
BUTTON_OFFICER_GPIO = 27

# init hardware buttons
btn_victim = Button(BUTTON_VICTIM_GPIO, pull_up=True)
btn_officer = Button(BUTTON_OFFICER_GPIO, pull_up=True)

# init VOSK model (download a small model to /home/pi/models/vosk-model-small)
model = Model("/home/pi/models/vosk-model-small-en-us")  # choose model per languages

def record_wav(filename, seconds=RECORD_SECONDS, rate=SAMPLE_RATE):
    print("Recording...")
    recording = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype='int16')
    sd.wait()
    wavfile.write(filename, rate, recording)
    print("Saved", filename)

def transcribe_vosk(wav_path):
    import wave, json
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            j = json.loads(rec.Result())
            text += " " + j.get("text", "")
    j = json.loads(rec.FinalResult())
    text += " " + j.get("text", "")
    return text.strip()

def detect_language(text):
    # wrapper uses fastText; returns language code like 'en' or 'es'
    return detect(text)

def elevenlabs_tts_and_play(text, voice_name="Rachel"):
    # Example SDK usage — check the version of SDK for exact API
    from elevenlabs import generate, play, set_api_key
    set_api_key(ELEVENLABS_API_KEY)
    audio_bytes = generate(text=text, voice=voice_name)  # SDK method may differ
    with open("/tmp/out.mp3", "wb") as f:
        f.write(audio_bytes)
    # play with an external command (mpg123) or with pydub
    os.system("mpg123 /tmp/out.mp3")

def handle_victim_press():
    path = "/tmp/victim.wav"
    record_wav(path, seconds=8)
    transcript = transcribe_vosk(path)
    print("Transcript:", transcript)
    lang = detect_language(transcript)
    print("Detected language:", lang)
    # If not English, either translate or produce English TTS
    # For simplicity, assume transcript is original and we want english TTS
    # You'd call a translation API here if needed.
    elevenlabs_tts_and_play(transcript, voice_name="alloy")  # choose voice

# Bind button events (simple blocking demo)
print("Ready. Press victim button.")
btn_victim.when_pressed = handle_victim_press

# Keep main loop
import time
while True:
    time.sleep(1)
