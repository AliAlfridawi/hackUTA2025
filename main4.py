#!/usr/bin/env python3
"""
Voice Translation: Whisper (STT) → Gemini (translate) → ElevenLabs (TTS)

Keys:
  v : record in ANY language → translate to English → speak in English
  o : record in English → translate back to previously detected language → speak in that language
  q : quit

Env Vars:
  ELEVEN_LABS_API_KEY   (required)
  GEMINI_API_KEY        (required)
  WHISPER_MODEL_NAME    (optional, default "small": tiny/base/small/medium/large-v3)
"""

import os
import sys
import tempfile
import requests
import pygame
import soundfile as sf
import speech_recognition as sr
from pynput import keyboard
from faster_whisper import WhisperModel
import google.generativeai as genai

# -----------------------------
# Config & Environment
# -----------------------------
ELEVEN_LABS_API_KEY = "sk_b825ad16a92c0ef7693965b5b60a69ddd1f8b33ac1642b6"
GEMINI_API_KEY = "AIzaSyA7Ybc1RmRl3VzngiRfRwqEgfC5E6WBAJw"
WHISPER_MODEL_NAME = "small"

# ElevenLabs voice ID (you can change this to any voice ID from your account)
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")

# If pygame audio fails on some systems, you can set:
# os.environ.setdefault("SDL_AUDIODRIVER", "dsp")

# Simple ISO-639-1 to language name map (extend if you like)
LANG_CODE_TO_NAME = {
    "en":"English","es":"Spanish","fr":"French","de":"German","it":"Italian","pt":"Portuguese",
    "ru":"Russian","zh":"Chinese","ja":"Japanese","ko":"Korean","ar":"Arabic","hi":"Hindi",
    "tr":"Turkish","vi":"Vietnamese","th":"Thai","pl":"Polish","nl":"Dutch","uk":"Ukrainian",
    "el":"Greek","he":"Hebrew","id":"Indonesian","sv":"Swedish","ro":"Romanian","cs":"Czech",
    "hu":"Hungarian","fi":"Finnish","bn":"Bengali","ta":"Tamil","te":"Telugu","ur":"Urdu",
}

def language_name_from_code(code: str) -> str:
    return LANG_CODE_TO_NAME.get((code or "").lower(), code or "Unknown")

# -----------------------------
# Sanity checks
# -----------------------------
if not ELEVEN_LABS_API_KEY:
    print("❌ ELEVEN_LABS_API_KEY not set. `export ELEVEN_LABS_API_KEY='...'`")
    sys.exit(1)

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not set. `export GEMINI_API_KEY='...'`")
    sys.exit(1)

# -----------------------------
# Initialize libraries
# -----------------------------
print("⏳ Loading Whisper model:", WHISPER_MODEL_NAME)
whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="auto", compute_type="auto")

print("⏳ Initializing Gemini (text-only)…")
genai.configure(api_key=GEMINI_API_KEY)
# Prefer fast, text-capable model. Fallback to gemini-pro if needed.
PREFERRED_GEMINI_MODELS = ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-pro"]
def build_gemini_model():
    try:
        models = {
            m.name.split("/", 1)[-1]: m
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        }
        for name in PREFERRED_GEMINI_MODELS:
            if name in models:
                print(f"✅ Using Gemini model: {name}")
                return genai.GenerativeModel(name), name
    except Exception as e:
        print(f"⚠️ Could not list models: {e}. Falling back to 'gemini-pro'.")
    print("⚠️ Falling back to: gemini-pro")
    return genai.GenerativeModel("gemini-pro"), "gemini-pro"

gemini_model, active_gemini_model_name = build_gemini_model()

# Audio playback
pygame.mixer.init()

# Mic capture
recognizer = sr.Recognizer()

# Tracks language detected in last 'v' press
detected_language_name = None  # e.g., "Spanish"
detected_language_code = None  # e.g., "es"

# -----------------------------
# Core functions
# -----------------------------
def record_audio() -> str | None:
    """Capture microphone audio to a temporary WAV file and return path."""
    print("🎤 Listening… Speak now!")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=12)
        except sr.WaitTimeoutError:
            print("❌ No speech detected. Try again.")
            return None
        except Exception as e:
            print(f"❌ Mic error: {e}")
            return None

    # Save to temp WAV
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio.get_wav_data())
            return f.name
    except Exception as e:
        print(f"❌ Could not write temp WAV: {e}")
        return None

def stt_whisper(audio_file_path: str) -> tuple[str | None, str | None]:
    """
    Transcribe + auto-detect language using faster-whisper.
    Returns: (lang_code, transcript) or (None, None)
    """
    try:
        # Ensure readable (optional, also validates file)
        _data, _sr = sf.read(audio_file_path)
        segments, info = whisper_model.transcribe(
            audio_file_path,
            vad_filter=True,
            beam_size=5,
            temperature=0.2,
        )
        transcript = "".join(s.text for s in segments).strip()
        lang_code = (info.language or "").lower()
        return (lang_code if transcript else None), (transcript or None)
    except Exception as e:
        print(f"❌ Whisper error: {e}")
        return None, None
    finally:
        # Clean temp file
        try:
            os.unlink(audio_file_path)
        except Exception:
            pass

def translate_text(text: str, source_lang_name: str, target_lang_name: str) -> str:
    """
    Translate text using Gemini (text → text).
    """
    prompt = (
        f"Translate the following text from {source_lang_name} to {target_lang_name}. "
        f"Respond with ONLY the translation, no extra words.\n\n"
        f"Text:\n{text}\n\nTranslation:"
    )
    try:
        resp = gemini_model.generate_content(prompt)
        out = (resp.text or "").strip()
        if not out:
            raise ValueError("Empty translation")
        return out
    except Exception as e:
        print(f"⚠️ Translation error ({active_gemini_model_name}): {e}")
        # Fallback: return original text so the flow continues
        return text

def text_to_speech(text: str):
    """
    Convert text to speech with ElevenLabs and play it.
    Uses the multilingual v2 model and the configured voice ID.
    """
    print("🔊 ElevenLabs TTS…")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_LABS_API_KEY,
    }
    body = {
        "text": text,
        "model_id": ELEVENLABS_TTS_MODEL,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }
    try:
        r = requests.post(url, json=body, headers=headers, timeout=60)
        if r.status_code != 200:
            print(f"❌ ElevenLabs error {r.status_code}: {r.text[:300]}")
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(r.content)
            mp3_path = f.name
        pygame.mixer.music.load(mp3_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(20)
        try:
            os.unlink(mp3_path)
        except Exception:
            pass
        print("✅ Spoke output.")
    except Exception as e:
        print(f"❌ TTS error: {e}")

# -----------------------------
# Keyboard flow
# -----------------------------
waiting_for_key = True

def handle_v_flow():
    """
    User speaks in ANY language → detect+transcribe → translate to English → speak English.
    """
    global detected_language_name, detected_language_code
    audio_path = record_audio()
    if not audio_path:
        return
    lang_code, original_text = stt_whisper(audio_path)
    if not (lang_code and original_text):
        print("❌ Could not transcribe/detect language. Try again.")
        return

    detected_language_code = lang_code
    detected_language_name = language_name_from_code(lang_code)
    print(f"🌐 Detected language: {detected_language_name} ({lang_code})")
    print(f"📝 Original ({detected_language_name}): {original_text}")

    translated = translate_text(original_text, detected_language_name, "English")
    print(f"✅ Translation (English): {translated}")
    text_to_speech(translated)

def handle_o_flow():
    """
    User speaks in English → translate back to the last detected language → speak it.
    """
    if not detected_language_name:
        print("❌ Press 'v' first to establish the other language.")
        return
    audio_path = record_audio()
    if not audio_path:
        return
    lang_code, english_text = stt_whisper(audio_path)
    if not english_text:
        print("❌ Could not transcribe English speech. Try again.")
        return

    # Even if Whisper mis-detects here, we *force* source as English for translation semantics.
    print(f"📝 Original (English): {english_text}")
    target = detected_language_name
    translated = translate_text(english_text, "English", target)
    print(f"✅ Translation ({target}): {translated}")
    text_to_speech(translated)

def on_press(key):
    global waiting_for_key
    if not waiting_for_key:
        return
    try:
        if hasattr(key, "char") and key.char == "v":
            waiting_for_key = False
            print("\n" + "="*56)
            print("🌍 MODE: Original language → English")
            print("="*56)
            handle_v_flow()
            print("\n💡 Press 'o' to respond in English, or 'v' to record again")
            waiting_for_key = True
        elif hasattr(key, "char") and key.char == "o":
            waiting_for_key = False
            print("\n" + "="*56)
            print("🇬🇧 MODE: English → Original language")
            print("="*56)
            handle_o_flow()
            print("\n💡 Press 'v' to record in original language, or 'o' to respond again")
            waiting_for_key = True
        elif hasattr(key, "char") and key.char == "q":
            print("\n👋 Exiting…")
            return False
    except AttributeError:
        pass

# -----------------------------
# Main
# -----------------------------
def main():
    print("="*56)
    print("🎙️  Voice Translation — Whisper + Gemini + ElevenLabs")
    print("="*56)
    print(f"✅ Gemini model: {active_gemini_model_name}")
    print("\nInstructions:")
    print("  • Press 'v' - Speak in any language → English")
    print("  • Press 'o' - Speak in English → back to original detected language")
    print("  • Press 'q' - Quit")
    print("\n⏳ Waiting for key press…\n")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()
