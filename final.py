# pi_translator_gpio.py
import os
import sys
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import subprocess
from dotenv import load_dotenv
from tqdm import tqdm

# GPIO / LEDs / Buttons
from gpiozero import Button, LED

# ElevenLabs + translation
try:
    from elevenlabs import ElevenLabs
except Exception:
    ElevenLabs = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

# -------------------- GPIO PINS (BCM) --------------------
BTN_PTT_GPIO = int(os.getenv("BTN_PTT_GPIO", "17"))   # push-to-talk (hold)
LED_RED_GPIO = int(os.getenv("LED_RED_GPIO", "23"))   # red during recording
LED_GREEN_GPIO = int(os.getenv("LED_GREEN_GPIO", "24"))  # green during process/play
BTN_BACK_ENV = os.getenv("BTN_BACK_GPIO")             # optional back-translate button
BTN_BACK_GPIO = int(BTN_BACK_ENV) if BTN_BACK_ENV not in (None, "", "None") else None

# -------------------- AUDIO SETTINGS --------------------
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))
DTYPE = np.float32

# Optional device selection (index or name)
SD_INPUT_DEVICE = os.getenv("SD_INPUT_DEVICE")
SD_OUTPUT_DEVICE = os.getenv("SD_OUTPUT_DEVICE")

def _parse_device(v):
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return v

_SD_INPUT_DEVICE = _parse_device(SD_INPUT_DEVICE)
_SD_OUTPUT_DEVICE = _parse_device(SD_OUTPUT_DEVICE)

def choose_output_device():
    """Return an output device index or name. Preference:
    - env override (_SD_OUTPUT_DEVICE)
    - USB devices (name contains 'usb')
    - 'headphone' or 'bcm' names
    - default output device
    """
    if _SD_OUTPUT_DEVICE is not None:
        print(f"Using SD_OUTPUT_DEVICE from env: {_SD_OUTPUT_DEVICE}")
        return _SD_OUTPUT_DEVICE
    try:
        devs = sd.query_devices()
        # prefer USB PnP or any device with 'usb' in its name
        for i, d in enumerate(devs):
            name = d.get('name','').lower()
            if d.get('max_output_channels', 0) > 0 and 'usb' in name:
                print(f"Auto-selected USB output device {i}: {d.get('name')}")
                return i
        # prefer bcm2835/headphones
        for i, d in enumerate(devs):
            name = d.get('name','').lower()
            if d.get('max_output_channels', 0) > 0 and ('headphone' in name or 'bcm' in name):
                print(f"Auto-selected headphone output device {i}: {d.get('name')}")
                return i
        # fallback to default output
        default_out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        print(f"Using default output device: {default_out}")
        return default_out
    except Exception as e:
        print(f"Could not query devices to auto-select output: {e}")
        return _SD_OUTPUT_DEVICE

# Pick the output device once at import-time
SELECTED_SD_OUTPUT_DEVICE = choose_output_device()

# Output gain (float). Values >1.0 increase volume. Keep modest to avoid clipping.
OUTPUT_GAIN = float(os.getenv('OUTPUT_GAIN', '10.0'))

def apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    """Apply gain to float audio in range [-1,1] with clipping protection."""
    if gain == 1.0 or audio is None:
        return audio
    out = audio * float(gain)
    # Clip safely
    out = np.clip(out, -1.0, 1.0)
    return out

# -------------------- LOAD KEYS / INIT CLIENTS --------------------
load_dotenv()

if ElevenLabs is None:
    print("❌ ElevenLabs SDK not installed. Install with: pip install elevenlabs")
    sys.exit(1)

ELEVEN_API = os.getenv("ELEVEN_API")
if not ELEVEN_API:
    print("❌ ELEVEN_API not set. Put it in .env or export it before running.")
    sys.exit(1)

try:
    elevenlabs = ElevenLabs(api_key=ELEVEN_API)
except Exception as e:
    print(f"❌ Error initializing ElevenLabs: {e}")
    sys.exit(1)

VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
tts_model_id = "eleven_multilingual_v2"

if GoogleTranslator is None:
    print("⚠️  deep-translator not installed; translation will be skipped.")
translator = GoogleTranslator(source='auto', target='en') if GoogleTranslator else None

# -------------------- STATE --------------------
recording = False
audio_data = []
processing_audio = False

# Debug helper: count first callback calls after recording starts
callback_debug_count = 0

last_source_text = None
last_source_lang = None
last_english_text = None

state_lock = threading.Lock()
stop_program = False

# -------------------- LEDs --------------------
led_red = LED(LED_RED_GPIO)
led_green = LED(LED_GREEN_GPIO)
led_red.off()
led_green.off()

# -------------------- HELPERS --------------------
def process_with_progress(description, total=100):
    return tqdm(total=total, desc=description, leave=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')

def map_sdk_lang_to_iso(sdk_code: str) -> str:
    if not sdk_code:
        return None
    sdk_code = sdk_code.lower()
    mapping = {
        'spa': 'es', 'eng': 'en', 'fra': 'fr', 'deu': 'de', 'ita': 'it',
        'por': 'pt', 'rus': 'ru', 'jpn': 'ja', 'kor': 'ko', 'zho': 'zh',
        'ara': 'ar', 'hin': 'hi'
    }
    return mapping.get(sdk_code, sdk_code[:2])

def likely_english(text: str) -> bool:
    """Simple heuristic to guess if text is English when language_code is missing.
    Counts common English function words; returns True if several are present.
    """
    if not text:
        return False
    text = text.lower()
    common = [' the ', ' and ', ' is ', ' i ', ' you ', ' to ', ' of ', ' that ', ' it ']
    count = 0
    for w in common:
        if w in f' {text} ':
            count += 1
    return count >= 2

def extract_text_from_transcription(transcription):
    if transcription is None:
        return None
    if isinstance(transcription, str):
        return transcription

    # dict-like
    if isinstance(transcription, dict):
        # common shapes: {"text": "..."} or {"transcript": "..."}
        return transcription.get("text") or transcription.get("transcript")

    # attribute style
    if hasattr(transcription, "text"):
        return getattr(transcription, "text")

    if hasattr(transcription, "transcripts"):
        try:
            t = getattr(transcription, "transcripts")
            if isinstance(t, (list, tuple)) and t:
                item = t[0]
                if hasattr(item, "text"):
                    return getattr(item, "text")
                if isinstance(item, dict):
                    return item.get("text")
        except Exception:
            pass

    try:
        return str(transcription)
    except Exception:
        return None

# -------------------- AUDIO LOOP --------------------
def audio_callback(indata, frames, time_info, status):
    global audio_data
    global callback_debug_count
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    if recording:
        buf = np.mean(indata, axis=1) if indata.ndim > 1 else indata
        audio_data.append(buf.copy())
        try:
            if callback_debug_count < 5:
                peak = float(np.max(np.abs(buf))) if buf.size else 0.0
                print(f"[audio_callback] frames={frames} peak={peak:.6f}")
                callback_debug_count += 1
        except Exception:
            pass

def audio_thread_loop():
    global stop_program
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=int(SAMPLE_RATE * 0.1),
            callback=audio_callback,
            device=_SD_INPUT_DEVICE
        ):
            print("✅ Audio stream ready. Hold the button to talk.")
            while not stop_program:
                time.sleep(0.05)
    except Exception as e:
        print(f"❌ Audio stream error: {e}")
        try:
            print("Available devices:\n", sd.query_devices())
        except Exception:
            pass

# -------------------- CORE PROCESSING --------------------
def save_and_process_audio():
    """Recorded audio -> WAV -> STT -> translate EN -> TTS -> playback."""
    global processing_audio, last_english_text, last_source_text, last_source_lang

    with state_lock:
        if not audio_data:
            print("⚠️  No audio data recorded.")
            return
        processing_audio = True

    print("\n📝 Processing your audio...")
    temp_wav = "temp_recording.wav"

    try:
        # Combine + normalize
        progress = process_with_progress("Preparing audio", 100)
        combined = np.concatenate(audio_data).astype(np.float32)
        if combined.size == 0:
            raise ValueError("Recorded audio is empty")

        max_val = float(np.max(np.abs(combined)))
        if max_val > 0:
            combined = combined / max_val

        # Save as WAV (reliable for STT)
        sf.write(temp_wav, combined, SAMPLE_RATE, subtype='PCM_16')

        for _ in range(100):
            progress.update(1); time.sleep(0.0025)
        progress.close()

        # STT (use file=..., file_format='other' to match SDKs commonly)
        progress = process_with_progress("Converting speech to text", 100)
        with open(temp_wav, "rb") as f:
            transcription = elevenlabs.speech_to_text.convert(
                file=f,
                model_id="scribe_v1",
                file_format="other"
            )
        progress.update(100); progress.close()

        detected_text = extract_text_from_transcription(transcription)
        if not detected_text or not str(detected_text).strip():
            print("❌ No speech detected. Try speaking closer to the mic.")
            return

        source_text = str(detected_text).strip()
        sdk_lang = getattr(transcription, 'language_code', None)
        detected_iso = map_sdk_lang_to_iso(sdk_lang) if sdk_lang else None
        if not detected_iso:
            # fallback to heuristic
            detected_iso = 'en' if likely_english(source_text) else 'unknown'
        print(f"\n🗣️  Detected ({detected_iso}): {source_text}")

        # If we have a previous (source) language and the current input is English,
        # treat this as a reply and translate it back into the previous language.
        is_reply_to_previous = False
        if last_source_lang and last_source_lang not in (None, '', 'en', 'unknown'):
            if detected_iso == 'en' or likely_english(source_text):
                is_reply_to_previous = True

        if is_reply_to_previous:
            # Translate English reply back to previous source language
            print(f"\n🔁 Back-translating reply to {last_source_lang}...")
            try:
                if translator:
                    back_text = GoogleTranslator(source='auto', target=last_source_lang).translate(source_text)
                else:
                    back_text = source_text
            except Exception as e:
                print(f"⚠️  Back-translation error: {e}")
                back_text = source_text

            print(f"🔤 Back text: {back_text}")

            with state_lock:
                # snapshot the reply
                last_english_text = source_text
                last_source_text = back_text

            # TTS -> play in previous language
            print("🎯 Generating back-translation speech...")
            progress = process_with_progress("Generating speech", 100)
            pcm_chunks = []
            for chunk in elevenlabs.text_to_speech.convert(
                VOICE_ID,
                text=back_text,
                output_format="pcm_16000",
                model_id=tts_model_id,
                language_code=last_source_lang
            ):
                pcm_chunks.append(chunk)
            progress.update(100); progress.close()
            if not pcm_chunks:
                print("❌ No audio generated from TTS")
                return
            # Play back the reply in the original speaker's language
            print("🔊 Playing back-translation...")
            # assemble and play reply pcm
            audio_bytes = b"".join(pcm_chunks)
            pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = (pcm16.astype(np.float32) / 32768.0).reshape(-1, 1)
            # apply gain and play
            audio_to_play = apply_gain(audio_float, OUTPUT_GAIN)
            try:
                sd.play(audio_to_play, 16000, device=SELECTED_SD_OUTPUT_DEVICE)
            except Exception:
                print('Playback with selected device failed, falling back to default')
                sd.play(audio_to_play, 16000)
            sd.wait()
            print("✅ Back-translation playback complete\n")
            # snapshot state done; we're finished for this input
            return
        # --- Normal path: translate source_text to English and speak it ---
        progress = process_with_progress("Translating to English", 100)
        try:
            if translator:
                english_text = translator.translate(source_text)
            else:
                english_text = source_text
        except Exception as e:
            print(f"⚠️  Translation error: {e}")
            english_text = source_text
        progress.update(100); progress.close()
        print(f"🔤 English: {english_text}")

        with state_lock:
            last_english_text = english_text
            last_source_text = source_text
            last_source_lang = detected_iso

        # TTS -> play (directly from PCM16 bytes)
        print("🎯 Generating English speech...")
        progress = process_with_progress("Generating speech", 100)
        pcm_chunks = []
        for chunk in elevenlabs.text_to_speech.convert(
            VOICE_ID,
            text=english_text,
            output_format="pcm_16000",
            model_id=tts_model_id
        ):
            pcm_chunks.append(chunk)
        progress.update(100); progress.close()

        if not pcm_chunks:
            print("❌ No audio generated from TTS")
            return

        # Join and play
        audio_bytes = b"".join(pcm_chunks)
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = (pcm16.astype(np.float32) / 32768.0).reshape(-1, 1)

        print("🔊 Playing translation...")
        # apply gain and play
        audio_to_play = apply_gain(audio_float, OUTPUT_GAIN)
        try:
            sd.play(audio_to_play, 16000, device=SELECTED_SD_OUTPUT_DEVICE)
        except Exception:
            print('Playback with selected device failed, falling back to default')
            sd.play(audio_to_play, 16000)
        sd.wait()
        print("✅ Playback complete\n")

    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        import traceback; traceback.print_exc()
        try:
            print("Available devices:\n", sd.query_devices())
        except Exception:
            pass
    finally:
        # clean temp
        try:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        except Exception:
            pass
        with state_lock:
            processing_audio = False

# -------------------- BACK TRANSLATION --------------------
BACK_RECORD_SECONDS = int(os.getenv("BACK_RECORD_SECONDS", "5"))

def translate_back():
    """Record English reply and translate back to the detected language -> TTS -> play."""
    with state_lock:
        target_lang = last_source_lang

    if not target_lang or target_lang == 'unknown':
        print("⚠️  No source language detected yet. First speak in a non-English language.")
        return

    print(f"\n🎤 Recording English reply for {BACK_RECORD_SECONDS} seconds...")
    try:
        rec = sd.rec(
            int(BACK_RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=_SD_INPUT_DEVICE
        )
        sd.wait()

        rec = np.mean(rec, axis=1) if rec.ndim > 1 else rec
        max_val = float(np.max(np.abs(rec))) if rec.size else 0.0
        if max_val > 0:
            rec = rec / max_val

        # Write wav for STT
        temp_reply = "temp_reply.wav"
        sf.write(temp_reply, rec.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")

        print("📝 Converting reply to text...")
        with open(temp_reply, "rb") as f:
            transcription = elevenlabs.speech_to_text.convert(
                file=f,
                model_id="scribe_v1",
                file_format="other"
            )
        reply_text = extract_text_from_transcription(transcription)
        if not reply_text or not reply_text.strip():
            print("❌ No speech detected in reply.")
            return
        reply_text = reply_text.strip()
        print(f"🗣️  Your reply: {reply_text}")

        print(f"🔄 Translating back to {target_lang}...")
        if GoogleTranslator:
            back_text = GoogleTranslator(source='auto', target=target_lang).translate(reply_text)
        else:
            back_text = reply_text
        print(f"🔤 Translation: {back_text}")

        print("🎯 Generating speech...")
        pcm_chunks = []
        for chunk in elevenlabs.text_to_speech.convert(
            VOICE_ID,
            text=back_text,
            output_format="pcm_16000",
            model_id=tts_model_id
        ):
            pcm_chunks.append(chunk)
        if not pcm_chunks:
            print("❌ No audio generated from TTS")
            return

        audio_bytes = b"".join(pcm_chunks)
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = (pcm16.astype(np.float32) / 32768.0).reshape(-1, 1)

        print("🔊 Playing back-translation...")
        # apply gain and play
        audio_to_play = apply_gain(audio_float, OUTPUT_GAIN)
        try:
            sd.play(audio_to_play, 16000, device=SELECTED_SD_OUTPUT_DEVICE)
        except Exception:
            print('Playback with selected device failed, falling back to default')
            sd.play(audio_to_play, 16000)
        sd.wait()
        print("✅ Back-translation complete\n")

    except Exception as e:
        print(f"❌ Error in back-translation: {e}")
        try:
            print("Available devices:\n", sd.query_devices())
        except Exception:
            pass
    finally:
        try:
            if os.path.exists("temp_reply.wav"):
                os.remove("temp_reply.wav")
        except Exception:
            pass

# -------------------- BUTTON EVENTS --------------------
def on_ptt_press():
    global recording, audio_data
    audio_data = []
    recording = True
    led_green.off()
    led_red.on()
    print("\n🎤 Recording... (release to stop)")

def on_ptt_release():
    global recording
    recording = False
    led_red.off()
    led_green.on()
    print("✋ Recording stopped. Processing...")
    threading.Thread(target=save_and_process_audio, daemon=True).start()

def on_back_press():
    threading.Thread(target=translate_back, daemon=True).start()

# -------------------- MAIN --------------------
def main():
    global stop_program
    print("\n" + "="*52)
    print("🎤 Voice Translator (Raspberry Pi + gpiozero)")
    print("="*52)

    # Quick audio preflight
    try:
        _ = sd.query_devices()
    except Exception as e:
        print("❌ Audio subsystem problem. Install PortAudio & reboot, or set SD_INPUT_DEVICE/SD_OUTPUT_DEVICE.")
        print(e)

    # Try to max out the system mixer volume so playback is loud
    def set_system_volume_max():
        cmds = [
            ["amixer", "sset", "Master", "100%", "unmute"],
            ["amixer", "sset", "PCM", "100%", "unmute"],
            ["amixer", "sset", "Headphone", "100%", "unmute"],
            ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"],
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "100%"],
        ]
        for cmd in cmds:
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Ran: {' '.join(cmd)}")
            except FileNotFoundError:
                # command not present on this system
                pass
            except subprocess.CalledProcessError as e:
                # continue trying other commands
                print(f"Volume command failed: {' '.join(cmd)} -> {e}")

    try:
        set_system_volume_max()
    except Exception as e:
        print(f"Could not set system volume: {e}")

    try:
        # Buttons
        btn = Button(BTN_PTT_GPIO, pull_up=True, bounce_time=0.05)
        btn.when_pressed = on_ptt_press
        btn.when_released = on_ptt_release

        if BTN_BACK_GPIO is not None:
            btn2 = Button(BTN_BACK_GPIO, pull_up=True, bounce_time=0.1)
            btn2.when_pressed = on_back_press
            print(f"🔁 Back-translation button enabled on GPIO {BTN_BACK_GPIO}")

        # Audio input thread
        th = threading.Thread(target=audio_thread_loop, daemon=True)
        th.start()

        print("\n📋 Instructions:")
        print("  • Hold the main button to record (RED LED on)")
        print("  • Release to process and hear translation (GREEN LED on)")
        if BTN_BACK_GPIO is not None:
            print("  • Press back button to record brief English reply and play back in source language")
        print("\nPress Ctrl+C to exit\n")

        while True:
            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback; traceback.print_exc()
    finally:
        stop_program = True
        led_red.off()
        led_green.off()
        print("✅ Cleanup complete. Goodbye!\n")

if __name__ == "__main__":
    main()
