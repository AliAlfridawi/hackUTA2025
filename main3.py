#!/usr/bin/env python3
"""
ElevenLabs Voice Translation Script
Press 'v' to record in any language -> translates to English
Press 'o' to record in English -> translates back to original language
Press 'q' to quit
"""

import os
import speech_recognition as sr
from pynput import keyboard
import tempfile
import pygame
import requests
import google.generativeai as genai

# Get API keys from environment variables
ELEVEN_LABS_API_KEY ="k_b825ad16a92c0ef7693965b5b60a69ddd1f8b33ac1642b6"
GEMINI_API_KEY ="AIzaSyA7Ybc1RmRl3VzngiRfRwqEgfC5E6WBAJw"

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use gemini-1.5-flash for audio support (faster and supports audio files)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# Initialize pygame mixer for playing audio
pygame.mixer.init()

# Initialize speech recognizer for capturing microphone input
recognizer = sr.Recognizer()

# Variable to store the detected language from the first input (when 'v' is pressed)
detected_language = None

# Variable to store the user's preferred language (for better recognition)
user_language = None

# Flag to track which mode we're in
waiting_for_key = True


def record_audio(expected_language=None):
    """
    Records audio from the microphone and saves it to a temporary file.
    Returns the audio file path for Gemini to process.
    
    Args:
        expected_language: Not used with Gemini (it auto-detects)
    
    Returns:
        Audio file path or None if recording failed
    """
    print("🎤 Listening... Speak now!")
    
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        # Adjust for ambient noise to improve recognition accuracy
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            # Listen for audio input - timeout after 5 seconds of silence
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("⏳ Recording complete, processing with Gemini...")
            
            # Save audio to a temporary WAV file for Gemini
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio.get_wav_data())
                audio_path = temp_audio.name
            
            return audio_path
            
        except sr.WaitTimeoutError:
            print("❌ No speech detected. Please try again.")
            return None
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
            return None


def transcribe_and_translate_audio(audio_file_path, translate_to="English"):
    """
    Uses Gemini to:
    1. Listen to the audio
    2. Detect the language
    3. Transcribe what was said
    4. Translate it to the target language
    
    All in ONE API call! Gemini can handle audio directly.
    
    Args:
        audio_file_path: Path to the audio file
        translate_to: Target language for translation (e.g., "English", "Spanish")
    
    Returns:
        Dictionary with:
        - original_text: What was said
        - original_language: Language detected
        - translated_text: Translation
        Or None if failed
    """
    if not gemini_model:
        print("❌ Gemini API not configured.")
        return None
    
    try:
        print(f"🔄 Gemini is processing audio (detect + transcribe + translate to {translate_to})...")
        
        # Upload the audio file to Gemini
        audio_file = genai.upload_file(path=audio_file_path)
        
        # Single prompt that does EVERYTHING
        prompt = f"""Listen to this audio file and do the following:

1. Detect what language is being spoken
2. Transcribe exactly what was said in the original language
3. Translate it to {translate_to}

Respond in this EXACT format (don't add extra text):
Language: [language name]
Original: [transcription in original language]
Translation: [translation in {translate_to}]"""
        
        # Generate response with audio - Gemini processes the audio directly!
        response = gemini_model.generate_content([prompt, audio_file])
        result = response.text.strip()
        
        # Clean up the temporary audio file
        os.unlink(audio_file_path)
        
        # Parse the response
        lines = result.split('\n')
        language = None
        original_text = None
        translated_text = None
        
        for line in lines:
            if line.startswith("Language:"):
                language = line.replace("Language:", "").strip()
            elif line.startswith("Original:"):
                original_text = line.replace("Original:", "").strip()
            elif line.startswith("Translation:"):
                translated_text = line.replace("Translation:", "").strip()
        
        if language and original_text and translated_text:
            print(f"🌐 Detected language: {language}")
            print(f"📝 Original ({language}): {original_text}")
            print(f"✅ Translation ({translate_to}): {translated_text}")
            
            return {
                "original_text": original_text,
                "original_language": language,
                "translated_text": translated_text
            }
        else:
            print("❌ Could not parse Gemini response properly")
            print(f"Response was: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Error with Gemini: {e}")
        # Clean up audio file if it exists
        if os.path.exists(audio_file_path):
            os.unlink(audio_file_path)
        return None
    except sr.WaitTimeoutError:
            print("❌ No speech detected. Please try again.")
            return None, None
    except sr.RequestError as e:
            print(f"❌ Error with speech recognition service: {e}")
            return None, None


def translate_text(text, source_lang, target_lang):
    """
    Translates text using Gemini (without audio).
    This is used as a backup or for text-only translation.
    
    Args:
        text: The text to translate
        source_lang: Source language name
        target_lang: Target language name
    
    Returns:
        Translated text string
    """
    if not gemini_model:
        print("❌ Gemini API not configured. Returning original text.")
        return text
    
    try:
        print(f"🔄 Translating from {source_lang} to {target_lang}...")
        
        # Simple translation prompt
        prompt = f"""Translate this text from {source_lang} to {target_lang}.
Only provide the translation, nothing else.

Text: {text}

Translation:"""
        
        response = gemini_model.generate_content(prompt)
        translated = response.text.strip()
        
        print(f"✅ Translation: {translated}")
        return translated
        
    except Exception as e:
        print(f"❌ Translation error: {e}")
        print("📝 Returning original text")
        return text


def get_language_name(lang_name):
    """
    Helper function for compatibility.
    Gemini already returns full language names, so just return as-is.
    """
    return lang_name


def text_to_speech(text, language_code="en"):
    """
    Converts text to speech using ElevenLabs API and plays it through speakers.
    Uses direct API calls instead of the client library to avoid compatibility issues.
    
    Args:
        text: The text to convert to speech
        language_code: Language code for the speech (e.g., 'en', 'es', 'fr')
    """
    print(f"🔊 Converting to speech...")
    
    try:
        # ElevenLabs API endpoint for text-to-speech
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
        
        # Request headers with API key
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVEN_LABS_API_KEY
        }
        
        # Request body with text and voice settings
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        # Make the API request
        response = requests.post(url, json=data, headers=headers)
        
        # Check if request was successful
        if response.status_code == 200:
            # Save the audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(response.content)
                temp_filename = temp_file.name
            
            # Play the audio file
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up the temporary file
            os.unlink(temp_filename)
            print("✅ Speech played successfully!")
        else:
            print(f"❌ Error from ElevenLabs API: {response.status_code}")
            print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"❌ Error generating speech: {e}")


def on_press(key):
    """
    Keyboard listener function that triggers actions when specific keys are pressed.
    """
    global detected_language, waiting_for_key, user_language
    
    if not waiting_for_key:
        return
    
    try:
        # Check if 'v' key was pressed (original language input)
        if hasattr(key, 'char') and key.char == 'v':
            waiting_for_key = False
            print("\n" + "="*50)
            print("🌍 MODE: Recording in original language...")
            print("="*50)
            
            # Record audio from microphone
            audio_path = record_audio()
            
            if audio_path:
                # Use Gemini to process audio: detect language + transcribe + translate to English
                result = transcribe_and_translate_audio(audio_path, translate_to="English")
                
                if result:
                    # Store the detected language for later use
                    detected_language = result["original_language"]
                    
                    # Get the translated text (already in English)
                    translated_text = result["translated_text"]
                    
                    print(f"🇬🇧 Speaking in English...")
                    
                    # Convert English translation to speech and play
                    text_to_speech(translated_text, "en")
            
            print("\n💡 Press 'o' to respond in English, or 'v' to record again")
            waiting_for_key = True
        
        # Check if 'o' key was pressed (English input, translate to original language)
        elif hasattr(key, 'char') and key.char == 'o':
            if detected_language is None:
                print("❌ Please press 'v' first to detect the original language!")
                return
            
            waiting_for_key = False
            print("\n" + "="*50)
            print("🇬🇧 MODE: Recording in English...")
            print("="*50)
            
            # Record audio from microphone
            audio_path = record_audio()
            
            if audio_path:
                # Use Gemini to process audio: transcribe English + translate to original language
                result = transcribe_and_translate_audio(audio_path, translate_to=detected_language)
                
                if result:
                    # Get the translated text (in original language)
                    translated_text = result["translated_text"]
                    
                    print(f"🌍 Speaking in {detected_language}...")
                    
                    # Convert translated text to speech in original language and play
                    text_to_speech(translated_text, detected_language)
            
            print("\n💡 Press 'v' to record in original language, or 'o' to respond again")
            waiting_for_key = True
        
        # Check if 'q' key was pressed (quit)
        elif hasattr(key, 'char') and key.char == 'q':
            print("\n👋 Exiting...")
            return False  # Stop the listener
            
    except AttributeError:
        # Handle special keys (like shift, ctrl, etc.)
        pass


def main():
    """
    Main function that starts the keyboard listener and waits for key presses.
    """
    print("="*50)
    print("🎙️  Voice Translation with Gemini & ElevenLabs")
    print("="*50)
    
    print("\n💡 Gemini automatically detects language from audio!")
    print("📢 Gemini does EVERYTHING: listens → detects language → transcribes → translates")
    
    print("\n📋 Instructions:")
    print("  • Press 'v' - Record in any language (translates to English)")
    print("  • Press 'o' - Record in English (translates to original language)")
    print("  • Press 'q' - Quit the program")
    print("\n⏳ Waiting for key press...\n")
    
    # Start listening for keyboard events
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    # Check if API keys are set
    if not ELEVEN_LABS_API_KEY:
        print("❌ Error: ELEVEN_LABS_API_KEY environment variable not set!")
        print("Please set it with: export ELEVEN_LABS_API_KEY='your-api-key'")
        exit(1)
    
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY environment variable not set!")
        print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        exit(1)
    
    print("✅ Using Gemini 1.5 Flash for audio processing & translation (FREE!)")
    main()
