import os
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import keyboard
from elevenlabs import ElevenLabs
import threading
import time
from dotenv import load_dotenv
from tqdm import tqdm
import sys
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv()

# Initialize ElevenLabs client
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVEN_API"))

# Use the latest multilingual model
tts_model_id = "eleven_multilingual_v2"

# Recording settings
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = np.float32
recording = False
audio_data = []
translator = GoogleTranslator(source='auto', target='en')

# State for reverse translation (English -> original language)
last_source_text = None        # the original text (e.g., Spanish) detected earlier
last_source_lang = None        # ISO code like 'es'
last_english_text = None      # English text produced by translation
processing_audio = False      # True while save_and_process_audio is running
ready_for_back = False        # True once last_* variables are set

# Lock to protect access to the shared state variables above
state_lock = threading.Lock()

# Voice selection (env or fallback)
VOICE_ID = os.getenv('VOICE_ID', '21m00Tcm4TlvDq8ikWAM')

def record_audio():
    """Record audio from microphone"""
    global recording, audio_data
    audio_data = []
    
    def audio_callback(indata, frames, time, status):
        if status:
            print('Error:', status)
        if recording:
            # Normalize the input data
            normalized = np.mean(indata, axis=1) if indata.ndim > 1 else indata
            audio_data.append(normalized.copy())
    
    # Start the recording stream
    with sd.InputStream(samplerate=SAMPLE_RATE, 
                       channels=CHANNELS, 
                       dtype=DTYPE,
                       blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
                       callback=audio_callback):
        print("Press 'R' to start/stop recording")
        while True:
            if keyboard.is_pressed('esc'):  # Exit program
                break
            time.sleep(0.1)

def process_with_progress(description, total=100):
    """Creates a progress bar for a processing step"""
    progress_bar = tqdm(total=total, desc=description, leave=True, 
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
    return progress_bar

def save_and_process_audio():
    """Save recorded audio to WAV and process it (STT -> translate -> TTS).
    This version auto-detects the input language and stores the English translation
    plus the detected source language for later use by translate_back().
    """
    global processing_audio, ready_for_back, last_english_text, last_source_text, last_source_lang
    if not audio_data:
        return

    # Mark processing
    with state_lock:
        processing_audio = True
        ready_for_back = False

    print("\nProcessing your audio...")

    # Combine audio chunks and show a short prepare progress
    progress = process_with_progress("Preparing audio", 100)
    combined_audio = np.concatenate(audio_data)
    for _ in range(100):
        progress.update(1)
        time.sleep(0.01)
    progress.close()

    temp_wav = "temp_recording.wav"
    try:
        combined_audio = combined_audio.astype(np.float32)
        if combined_audio.size == 0:
            raise ValueError("Recorded audio is empty")
        max_val = float(np.max(np.abs(combined_audio)))
        if max_val > 0:
            combined_audio = combined_audio / max_val

        pcm_audio = np.int16(np.clip(combined_audio, -1.0, 1.0) * 32767)
        wavfile.write(temp_wav, SAMPLE_RATE, pcm_audio)
        duration = len(pcm_audio) / SAMPLE_RATE
        print(f"Saved temporary WAV: {temp_wav} | duration={duration:.2f}s | sample_rate={SAMPLE_RATE} | dtype={pcm_audio.dtype}")
    except Exception as e:
        progress.close()
        print(f"Error saving WAV: {e}")
        with state_lock:
            processing_audio = False
        return

    try:
        sys.stdout.flush()

        # STT (auto-detect language)
        progress = process_with_progress("Converting speech to text", 100)
        with open(temp_wav, 'rb') as audio_file:
            transcription = elevenlabs.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                file_format="other",
            )
        progress.update(100)
        progress.close()

        # Debug print
        try:
            print("DEBUG: transcription raw response:")
            import pprint
            pprint.pprint(transcription)
        except Exception:
            print(repr(transcription))

        detected_text = extract_text_from_transcription(transcription)
        if detected_text is None or not str(detected_text).strip():
            print("❌ No speech detected. Please try again.")
            with state_lock:
                processing_audio = False
            return

        source_text = str(detected_text).strip()
        sdk_lang = getattr(transcription, 'language_code', None)
        detected_iso = map_sdk_lang_to_iso(sdk_lang)

        print("\n📝 Detected Speech:")
        print("------------------------")
        print(f"🌐 Detected language (sdk): {sdk_lang} -> iso: {detected_iso}")
        print(f"🗣️  Source: {source_text}")
        sys.stdout.flush()

        # Translate to English (keep source_text variable)
        progress = process_with_progress("Translating to English", 100)
        try:
            english_text = translator.translate(source_text)
        except Exception as e:
            print(f"❌ Translation error: {e}")
            english_text = source_text
        progress.update(100)
        progress.close()

        # Snapshot translated English and source info under lock
        with state_lock:
            last_english_text = english_text
            last_source_text = source_text
            try:
                last_source_lang = map_sdk_lang_to_iso(sdk_lang)
            except Exception:
                last_source_lang = None
            ready_for_back = bool(last_english_text and last_source_lang)
            # allow translate_back to run while TTS is generated
            processing_audio = False

        print("\n📝 Translation Result:")
        print("------------------------")
        print(f"🔤  English: {english_text}")
        print("------------------------")
        sys.stdout.flush()

        # Generate English TTS
        print("\n🎯 Converting to speech...")
        progress = process_with_progress("Generating English speech", 100)
        audio_chunks = []
        try:
            for chunk in elevenlabs.text_to_speech.convert(
                VOICE_ID,
                text=english_text,
                output_format="pcm_16000",
                model_id=tts_model_id,
            ):
                audio_chunks.append(chunk)
        except Exception as e:
            progress.close()
            raise
        progress.update(100)
        progress.close()
        sys.stdout.flush()

        # Assemble and play TTS audio
        try:
            if not audio_chunks:
                raise RuntimeError('No audio data received from TTS')
            audio_bytes = b"".join(audio_chunks)
            import soundfile as sf
            with open('temp_response.raw', 'wb') as f:
                f.write(audio_bytes)
            data, samplerate = sf.read('temp_response.raw', samplerate=16000, channels=1, format='RAW', subtype='PCM_16')
            sf.write('temp_response.wav', data, samplerate)

            print("\n🔊 Playing translation...")
            data, samplerate = sf.read('temp_response.wav')
            sd.play(data, samplerate)
            sd.wait()
            print("✅ Playback complete\n")
        except Exception as e:
            print(f"❌ Error playing audio: {str(e)}")

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
    finally:
        # Clean up temp files
        try:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        except Exception:
            pass
        try:
            if os.path.exists("temp_response.wav"):
                os.remove("temp_response.wav")
        except Exception:
            pass

        # final safety: ensure processing flag cleared
        with state_lock:
            processing_audio = False

        if ready_for_back:
            print("Ready for back-translation — press 'E' to translate English back to the original language.")

def toggle_recording():
    """Toggle recording state"""
    global recording
    recording = not recording
    if recording:
        print("\n🎤 Recording started... (Press 'R' to stop)")
    else:
        print("\n✋ Recording stopped")
        save_and_process_audio()

def extract_text_from_transcription(transcription):
    """Return the transcribed text string from various transcription response shapes."""
    if transcription is None:
        return None
    # If it's already a string
    if isinstance(transcription, str):
        return transcription
    # If it's a mapping/dict-like
    try:
        from collections.abc import Mapping
        if isinstance(transcription, Mapping):
            return transcription.get("text") or transcription.get("transcript")
    except Exception:
        pass
    # Some SDK returns objects with attribute 'text'
    if hasattr(transcription, 'text'):
        return getattr(transcription, 'text')
    # Some responses wrap text under 'transcripts' or similar
    if hasattr(transcription, 'transcripts'):
        try:
            t = getattr(transcription, 'transcripts')
            if isinstance(t, (list, tuple)) and len(t) > 0:
                item = t[0]
                return getattr(item, 'text', None) or (item.get('text') if isinstance(item, dict) else None)
        except Exception:
            pass
    # Fallback: try to string-convert
    try:
        return str(transcription)
    except Exception:
        return None

def map_sdk_lang_to_iso(sdk_code: str) -> str:
    """Map SDK language codes like 'spa' to ISO 639-1 'es'."""
    if not sdk_code:
        return None
    sdk_code = sdk_code.lower()
    mapping = {
        'spa': 'es',
        'eng': 'en',
        'fra': 'fr',
        'deu': 'de',
        'ita': 'it',
        'por': 'pt',
        'rus': 'ru',
        # add more if needed
    }
    return mapping.get(sdk_code, sdk_code[:2])

# Debounce state for 'e' presses
_last_e_time = 0.0
_E_DEBOUNCE_SEC = 0.7
# How many seconds to record the second speaker's English reply when pressing 'E'
BACK_RECORD_SECONDS = 5

def translate_back():
    """Record a short English reply from the second speaker, transcribe it as English,
    translate into the previously-detected language, synthesize and play it.
    Bound to key 'e'."""
    global _last_e_time
    now = time.time()
    if now - _last_e_time < _E_DEBOUNCE_SEC:
        return
    _last_e_time = now

    # Snapshot required state
    with state_lock:
        if processing_audio:
            print("Still processing the recording — please wait a moment before pressing 'E'.")
            return
        target_lang = last_source_lang

    if not target_lang:
        print("No source language detected yet — use 'R' to record the first speaker first.")
        return

    print(f"Recording English reply for {BACK_RECORD_SECONDS} seconds... Speak now.")
    # Record a short clip for the second speaker
    try:
        rec = sd.rec(int(BACK_RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
        sd.wait()
        # Flatten if stereo
        recorded = np.mean(rec, axis=1) if rec.ndim > 1 else rec
        # Normalize and convert to PCM16
        recorded = recorded.astype(np.float32)
        max_val = float(np.max(np.abs(recorded))) if recorded.size > 0 else 0.0
        if max_val > 0:
            recorded = recorded / max_val
        pcm_audio = np.int16(np.clip(recorded, -1.0, 1.0) * 32767)
        temp_reply_wav = 'temp_reply.wav'
        wavfile.write(temp_reply_wav, SAMPLE_RATE, pcm_audio)
        print(f"Saved temporary reply WAV: {temp_reply_wav} | duration={BACK_RECORD_SECONDS}s")
    except Exception as e:
        print(f"Error recording reply: {e}")
        return

    # Transcribe the English reply
    print("Converting reply to text (English)...")
    progress = process_with_progress("Converting reply to text", 100)
    try:
        with open(temp_reply_wav, 'rb') as f:
            transcription = elevenlabs.speech_to_text.convert(
                file=f,
                model_id="scribe_v1",
                language_code="eng",
                file_format="other"
            )
        progress.update(100)
        progress.close()
    except Exception as e:
        progress.close()
        print(f"Error transcribing reply: {e}")
        if os.path.exists(temp_reply_wav):
            os.remove(temp_reply_wav)
        return

    reply_text = extract_text_from_transcription(transcription)
    if not reply_text or not str(reply_text).strip():
        print("No English speech detected in reply. Try again.")
        if os.path.exists(temp_reply_wav):
            os.remove(temp_reply_wav)
        return
    reply_text = str(reply_text).strip()
    print(f"🗣️  Detected English reply: {reply_text}")

    # Translate English reply into the previously-detected language
    print(f"Translating English reply back to {target_lang}...")
    progress = process_with_progress("Translating reply", 100)
    try:
        back_translator = GoogleTranslator(source='auto', target=target_lang)
        back_text = back_translator.translate(reply_text)
    except Exception as e:
        progress.close()
        print(f"Error translating reply: {e}")
        if os.path.exists(temp_reply_wav):
            os.remove(temp_reply_wav)
        return
    progress.update(100)
    progress.close()

    print("📝 Back-translation Result:")
    print("------------------------")
    print(f"🔤  Back: {back_text}")
    print("------------------------")

    # Synthesize the back-translated text and play it
    print("Generating speech for back-translation...")
    progress = process_with_progress("Generating back speech", 100)
    try:
        audio_chunks = []
        for chunk in elevenlabs.text_to_speech.convert(
            VOICE_ID,
            text=back_text,
            output_format="pcm_16000",
            model_id=tts_model_id,
            language_code=target_lang,
        ):
            audio_chunks.append(chunk)
    except Exception as e:
        progress.close()
        print(f"TTS for back-translation failed: {e}")
        if os.path.exists(temp_reply_wav):
            os.remove(temp_reply_wav)
        return
    progress.update(100)
    progress.close()

    try:
        if not audio_chunks:
            print("No audio data returned for back-translation")
            return
        audio_bytes = b"".join(audio_chunks)
        import soundfile as sf
        with open('temp_back_response.raw', 'wb') as f:
            f.write(audio_bytes)
        data, samplerate = sf.read('temp_back_response.raw', samplerate=16000, channels=1, format='RAW', subtype='PCM_16')
        sf.write('temp_back_response.wav', data, samplerate)

        print("\n🔊 Playing back-translation...")
        data, samplerate = sf.read('temp_back_response.wav')
        sd.play(data, samplerate)
        sd.wait()
        print("✅ Back-translation playback complete\n")
    except Exception as e:
        print(f"Error playing back-translation audio: {e}")
    finally:
        if os.path.exists('temp_back_response.raw'):
            os.remove('temp_back_response.raw')
        if os.path.exists('temp_back_response.wav'):
            os.remove('temp_back_response.wav')
        if os.path.exists(temp_reply_wav):
            os.remove(temp_reply_wav)

# Bind 'e' key to translate_back action on release (helps prevent repeated triggers)
keyboard.on_release_key('e', lambda _: translate_back())

# Main execution
def main():
    # Set up keyboard listener for recording toggle on release to avoid double-triggers
    keyboard.on_release_key('r', lambda _: toggle_recording())
    
    # Start recording thread
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()
    
    # Wait for the recording thread to finish
    record_thread.join()

if __name__ == "__main__":
    print("\n🎤 Voice Translator Started 🔊")
    print("================================")
    print("Controls:")
    print("  Press 'R' - Start/Stop recording")
    print("  Press 'ESC' - Exit program")
    print("================================")
    main()