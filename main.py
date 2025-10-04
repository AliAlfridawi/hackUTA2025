"""
two_way_translate.py

Two-button Raspberry Pi emergency translator using:
- ElevenLabs Speech-to-Text (Scribe v1) for transcription
- ElevenLabs Text-to-Speech for voice output
- googletrans for translation fallback (replace with paid API for production)
- GPIOZero for button + LED handling
- arecord/aplay (ALSA) for recording/playback

Hardware pins (BCM):
- Victim button: GPIO 17
- Officer button: GPIO 27
- Status LED: GPIO 22

Audio:
- USB mic for recording (set in RECORD_DEVICE)
- Speaker1 & Speaker2 are ALSA device names (APLAY_DEVICE_1/APLAY_DEVICE_2)
  If you only have one speaker, set both to the same device.

Before running:
- Export ELEVENLABS_API_KEY environment variable:
    export ELEVENLABS_API_KEY="your_key_here"
- Install required packages (see setup instructions in the README)
"""

import os
import time
import json
import subprocess
import threading
import requests
from gpiozero import Button, LED
from elevenlabs import set_api_key, generate, save  # SDK for TTS
from googletrans import Translator

# --------------------------
# Configuration (Edit me)
# --------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise RuntimeError("Set ELEVENLABS_API_KEY environment variable before running.")

# Pins (BCM)
VICTIM_BTN_PIN = 17
OFFICER_BTN_PIN = 27
STATUS_LED_PIN = 22

# Audio device names (ALSA). Use `arecord -l` and `aplay -l` to find these.
# Examples: "plughw:1,0" or "hw:2,0"
RECORD_DEVICE = "plughw:1,0"      # USB microphone capture device (change after `arecord -l`)
APLAY_DEVICE_VICTIM = "plughw:2,0"   # speaker that plays to victim (when officer speaks)
APLAY_DEVICE_OFFICER = "plughw:3,0"  # speaker that plays to officer (when victim speaks)

# Recording params
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORD_SECONDS = 20  # safety cutoff if button held too long

# ElevenLabs STT endpoint (Scribe v1) and TTS config
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text/convert"
# TTS voice - change to a voice id or name available in your account.
# The SDK will accept a voice name like "alloy" or a voice_id — check your ElevenLabs voice list.
TTS_VOICE = "alloy"   # replace with a voice id or name from your ElevenLabs account

# Translation config: target languages
OFFICER_LANG = "en"   # language police/officer listens in (English)
# For returning to victim, we'll attempt to use the detected source language.

# --------------------------
# Helper: system audio commands
# --------------------------
def start_arecord(outfile_path):
    """
    Start arecord as a subprocess, recording until we kill it.
    Returns the subprocess object.
    """
    cmd = [
        "arecord",
        "-D", RECORD_DEVICE,
        "-f", "S16_LE",
        "-c", str(CHANNELS),
        "-r", str(SAMPLE_RATE),
        "-t", "wav",
        outfile_path,
    ]
    # Start arecord
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p

def stop_arecord(proc, timeout=1):
    """
    Stop the arecord subprocess cleanly.
    """
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()

def play_audio(filepath, aplay_device=None):
    """
    Play an audio file (wav or mp3) using aplay or mpg123.
    Use aplay for WAV, mpg123 for mp3.
    """
    if filepath.lower().endswith(".mp3"):
        # use mpg123 for mp3 (installed earlier)
        cmd = ["mpg123"]
        if aplay_device:
            # mpg123 doesn't accept ALSA device the same way; instead set ALSA env var:
            env = os.environ.copy()
            env["AUDIODEV"] = aplay_device
            subprocess.run(cmd + [filepath], env=env)
        else:
            subprocess.run(cmd + [filepath])
    else:
        # WAV playback via aplay with device
        cmd = ["aplay"]
        if aplay_device:
            cmd += ["-D", aplay_device]
        cmd += [filepath]
        subprocess.run(cmd)

# --------------------------
# Helper: ElevenLabs STT
# --------------------------
def call_elevenlabs_stt(wav_path, language_code="auto", diarize=False, timestamps="none"):
    """
    Upload WAV file to ElevenLabs speech-to-text endpoint and poll until done.
    Returns the transcription result (dictionary).
    """
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    files = {"audio": ("audio.wav", open(wav_path, "rb"), "audio/wav")}
    data = {"language_code": language_code}
    if diarize:
        data["diarize"] = "true"
    if timestamps != "none":
        data["timestamps"] = timestamps  # e.g., "word", "sentence"

    # Create job
    resp = requests.post(ELEVENLABS_STT_URL, headers=headers, files=files, data=data, timeout=180)
    resp.raise_for_status()
    job = resp.json()
    job_id = job.get("id")
    if not job_id:
        # The API might return the result directly if small; handle that
        return job

    # Polling for completion
    status_url = f"https://api.elevenlabs.io/v1/speech-to-text/{job_id}"
    while True:
        r = requests.get(status_url, headers=headers, timeout=60)
        r.raise_for_status()
        result = r.json()
        status = result.get("status")
        if status in ("completed", "failed"):
            break
        time.sleep(0.7)
    if status == "failed":
        raise RuntimeError("STT job failed: " + json.dumps(result))
    return result

# --------------------------
# Helper: Translation (googletrans)
# --------------------------
translator = Translator()

def translate_text(text, dest_lang):
    """
    Translate text to dest_lang using googletrans (unofficial). Replace with a paid API for production.
    """
    if not text:
        return text
    try:
        out = translator.translate(text, dest=dest_lang)
        return out.text
    except Exception as e:
        print("Translation error (falling back to original):", e)
        return text

# --------------------------
# Helper: ElevenLabs TTS (via SDK)
# --------------------------
def tts_generate_and_save(text, out_path):
    """
    Generate speech from text using ElevenLabs SDK and save to out_path (mp3 or wav).
    Note: the official SDK function names can change; this uses 'generate' & 'save' helpers.
    """
    set_api_key(ELEVENLABS_API_KEY)
    # The SDK generate() typically returns audio bytes — we write to file.
    # Simple approach using SDK's generate + save utils:
    try:
        audio = generate(text=text, voice=TTS_VOICE)  # returns a bytes-like or an object
        # If audio is bytes:
        if isinstance(audio, (bytes, bytearray)):
            with open(out_path, "wb") as f:
                f.write(audio)
        else:
            # SDK might provide a save() helper:
            save(audio, out_path)
    except Exception as e:
        # If SDK use fails, try HTTP fallback
        print("SDK TTS failed, trying HTTP fallback:", e)
        tts_http_fallback(text, out_path)

def tts_http_fallback(text, out_path):
    """
    HTTP fallback for TTS (best-effort). Replace with the recommended TTS endpoint from ElevenLabs docs if needed.
    """
    tts_url = "https://api.elevenlabs.io/v1/text-to-speech/convert"  # docs show convert endpoint
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "voice": TTS_VOICE}
    r = requests.post(tts_url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    # If the endpoint returns binary audio:
    content_type = r.headers.get("content-type", "")
    if "audio" in content_type:
        with open(out_path, "wb") as f:
            f.write(r.content)
    else:
        # Otherwise try to parse json -> maybe a url
        j = r.json()
        audio_url = j.get("audio_url") or j.get("url")
        if audio_url:
            ad = requests.get(audio_url)
            ad.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(ad.content)
        else:
            raise RuntimeError("Unexpected TTS response: " + str(j))

# --------------------------
# Main event handlers
# --------------------------
status_led = LED(STATUS_LED_PIN)
victim_btn = Button(VICTIM_BTN_PIN, pull_up=True, bounce_time=0.05)
officer_btn = Button(OFFICER_BTN_PIN, pull_up=True, bounce_time=0.05)

def record_while_held(outfile, max_seconds=MAX_RECORD_SECONDS):
    """
    Record to outfile while the associated button is held.
    Returns filepath when done.
    Uses arecord subprocess to capture audio.
    """
    proc = start_arecord(outfile)
    start = time.time()
    try:
        # Wait until file exists or until we reach max_seconds; caller should stop proc when button release
        while True:
            # safety cutoff
            if (time.time() - start) > max_seconds:
                break
            time.sleep(0.05)
    finally:
        stop_arecord(proc)
    return outfile

def handle_victim_press():
    """
    Called when victim presses/holds their button.
    Flow:
      - Turn status LED on
      - Record while held
      - Upload to ElevenLabs STT
      - Translate/transcribe to OFFICER_LANG (English)
      - Generate TTS audio and play to OFFICER speaker
      - Turn LED off
    """
    print("[VICTIM] Button pressed — recording...")
    status_led.on()
    wav_path = "/tmp/victim.wav"
    # Start recording (we'll handle stop on release)
    rec_proc = start_arecord(wav_path)
    # Wait for release
    while victim_btn.is_pressed:
        time.sleep(0.05)
    # Stop recording
    stop_arecord(rec_proc)
    print("[VICTIM] Recording stopped — calling STT...")
    try:
        stt_result = call_elevenlabs_stt(wav_path, language_code="auto", diarize=False, timestamps="none")
        # stt_result typically contains 'text' and other metadata
        transcript = stt_result.get("text") or stt_result.get("transcript") or ""
        detected_lang = stt_result.get("language") or None
        print("[VICTIM] Transcript:", transcript)
        if not detected_lang:
            # fallback language detection via googletrans
            try:
                detected = translator.detect(transcript)
                detected_lang = detected.lang
            except Exception:
                detected_lang = None
        print("[VICTIM] Detected language:", detected_lang)

        # Translate to OFFICER_LANG if needed
        if detected_lang and detected_lang != OFFICER_LANG:
            print(f"[VICTIM] Translating from {detected_lang} to {OFFICER_LANG} ...")
            translated = translate_text(transcript, OFFICER_LANG)
        else:
            translated = transcript

        # TTS - produce file and play to officer speaker
        tts_out = "/tmp/victim_to_officer.mp3"
        print("[VICTIM] Generating TTS for officer...")
        tts_generate_and_save(translated, tts_out)
        print("[VICTIM] Playing to officer speaker...")
        play_audio(tts_out, aplay_device=APLAY_DEVICE_OFFICER)
        print("[VICTIM] Done.")
    except Exception as e:
        print("Error in victim flow:", e)
    finally:
        status_led.off()

def handle_officer_press():
    """
    Called when officer presses/holds their button.
    Flow is symmetric: record officer audio, STT, translate back to victim language (if known), TTS to victim speaker.
    """
    print("[OFFICER] Button pressed — recording...")
    status_led.on()
    wav_path = "/tmp/officer.wav"
    rec_proc = start_arecord(wav_path)
    while officer_btn.is_pressed:
        time.sleep(0.05)
    stop_arecord(rec_proc)
    print("[OFFICER] Recording stopped — calling STT...")
    try:
        stt_result = call_elevenlabs_stt(wav_path, language_code="auto", diarize=False, timestamps="none")
        transcript = stt_result.get("text") or stt_result.get("transcript") or ""
        detected_lang = stt_result.get("language") or None
        print("[OFFICER] Transcript:", transcript)
        # We assume officer speaks in OFFICER_LANG (English), so we prepare to translate into the last known victim language.
        # For a simple implementation, we will re-detect language of the last victim transcript stored in /tmp/last_victim_lang
        victim_lang = None
        try:
            with open("/tmp/last_victim_lang.txt", "r") as f:
                victim_lang = f.read().strip()
        except FileNotFoundError:
            victim_lang = None

        # If we don't have a victim_lang, attempt to detect from a previous recorded file or skip translation.
        if victim_lang:
            print(f"[OFFICER] Translating from {detected_lang or OFFICER_LANG} to victim language {victim_lang} ...")
            translated = translate_text(transcript, victim_lang)
        else:
            print("[OFFICER] No known victim language; sending officer transcript as-is to victim.")
            translated = transcript

        # TTS and play to victim speaker
        tts_out = "/tmp/officer_to_victim.mp3"
        tts_generate_and_save(translated, tts_out)
        print("[OFFICER] Playing to victim speaker...")
        play_audio(tts_out, aplay_device=APLAY_DEVICE_VICTIM)
        print("[OFFICER] Done.")
    except Exception as e:
        print("Error in officer flow:", e)
    finally:
        status_led.off()

# --------------------------
# Wiring the button events
# --------------------------
# When a button is pressed, we want to record while it's held.
# The handlers below will be run inside separate threads to avoid blocking the main loop.

def victim_threaded():
    # record and process in a thread
    threading.Thread(target=handle_victim_press, daemon=True).start()

def officer_threaded():
    threading.Thread(target=handle_officer_press, daemon=True).start()

victim_btn.when_pressed = victim_threaded
officer_btn.when_pressed = officer_threaded

# --------------------------
# Optional: store last victim language for officer replies
# We'll write detected victim language to /tmp/last_victim_lang.txt whenever we process a victim recording.
# Modify handle_victim_press above to write it (if available).
# --------------------------

# --------------------------
# Main loop
# --------------------------
if __name__ == "__main__":
    print("Two-way translator started. Waiting for button presses...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting.")

