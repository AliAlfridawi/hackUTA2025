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

# Get API key from environment variable
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

# Initialize pygame mixer for playing audio
pygame.mixer.init()

# Initialize speech recognizer for capturing microphone input
recognizer = sr.Recognizer()

# LibreTranslate API endpoint (free, no key needed)
LIBRETRANSLATE_URL = "https://libretranslate.com/translate"

# Variable to store the detected language from the first input (when 'v' is pressed)
detected_language = None

# Variable to store the user's preferred language (for better recognition)
user_language = None

# Flag to track which mode we're in
waiting_for_key = True


def record_audio(expected_language=None):
    """
    Records audio from the microphone until silence is detected.
    Returns the recognized text and detected language.
    
    Args:
        expected_language: If provided, only tries this language for faster/better recognition
    """
    print("🎤 Listening... Speak now!")
    
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        # Adjust for ambient noise to improve recognition accuracy
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            # Listen for audio input - timeout after 5 seconds of silence
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("⏳ Processing speech...")
            
            # Dictionary of supported languages
            # Format: language_code -> language_name
            supported_languages = {
                "en-US": "English",
                "es-ES": "Spanish",
                "fr-FR": "French",
                "de-DE": "German",
                "it-IT": "Italian",
                "pt-BR": "Portuguese",
                "ru-RU": "Russian",
                "ja-JP": "Japanese",
                "ko-KR": "Korean",
                "zh-CN": "Chinese (Mandarin)",
                "hi-IN": "Hindi",
                "ar-SA": "Arabic",
                "nl-NL": "Dutch",
                "pl-PL": "Polish",
                "tr-TR": "Turkish",
            }
            
            # If we know what language to expect, try only that one
            if expected_language and expected_language in supported_languages:
                try:
                    text = recognizer.recognize_google(audio, language=expected_language)
                    lang_name = supported_languages[expected_language]
                    print(f"📝 Recognized ({lang_name}): {text}")
                    return text, expected_language
                except sr.UnknownValueError:
                    print(f"❌ Could not understand audio as {supported_languages[expected_language]}")
                    return None, None
                except sr.RequestError as e:
                    print(f"❌ Error with speech recognition service: {e}")
                    return None, None
            
            # Otherwise, try all languages (slower but more flexible)
            # Try most common languages first for better performance
            for lang_code, lang_name in supported_languages.items():
                try:
                    # Attempt to recognize speech in this language
                    text = recognizer.recognize_google(audio, language=lang_code)
                    print(f"📝 Recognized ({lang_name}): {text}")
                    return text, lang_code
                except sr.UnknownValueError:
                    # This language didn't work, try the next one
                    continue
                except sr.RequestError as e:
                    print(f"❌ Error with speech recognition service: {e}")
                    return None, None
            
            # If none of the languages worked
            print("❌ Could not understand audio in any supported language.")
            print("💡 Try speaking more clearly or closer to the microphone")
            return None, None
            
        except sr.WaitTimeoutError:
            print("❌ No speech detected. Please try again.")
            return None, None
        except sr.RequestError as e:
            print(f"❌ Error with speech recognition service: {e}")
            return None, None


def translate_text(text, source_lang, target_lang):
    """
    Translates text from source language to target language using LibreTranslate API.
    LibreTranslate is free and open source - no API key needed!
    
    Args:
        text: The text to translate
        source_lang: Source language code (e.g., 'es', 'fr', 'de')
        target_lang: Target language code (e.g., 'en', 'es', 'fr')
    
    Returns:
        Translated text string
    """
    try:
        print(f"🔄 Translating from {source_lang} to {target_lang}...")
        
        # Make request to LibreTranslate API
        response = requests.post(
            LIBRETRANSLATE_URL,
            data={
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text"
            }
        )
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            translated = result.get("translatedText", text)
            print(f"✅ Translation: {translated}")
            return translated
        else:
            print(f"❌ Translation error: {response.status_code}")
            print("📝 Returning original text")
            return text
        
    except Exception as e:
        print(f"❌ Translation error: {e}")
        print("📝 Returning original text")
        return text


def convert_to_libretranslate_code(speech_lang_code):
    """
    Converts Google Speech Recognition language codes to LibreTranslate language codes.
    LibreTranslate uses simple 2-letter codes (ISO 639-1).
    
    Args:
        speech_lang_code: Language code from speech recognition (e.g., 'es-ES', 'fr-FR')
    
    Returns:
        LibreTranslate-compatible language code (e.g., 'es', 'fr', 'de')
    """
    # Mapping of speech recognition codes to LibreTranslate codes
    lang_map = {
        "en-US": "en",
        "en-GB": "en",
        "es-ES": "es",
        "fr-FR": "fr",
        "de-DE": "de",
        "it-IT": "it",
        "pt-BR": "pt",
        "pt-PT": "pt",
        "ru-RU": "ru",
        "ja-JP": "ja",
        "ko-KR": "ko",
        "zh-CN": "zh",
        "nl-NL": "nl",
        "pl-PL": "pl",
        "tr-TR": "tr",
        "ar-SA": "ar",
        "hi-IN": "hi",
    }
    
    # Return mapped code or try to extract base language
    if speech_lang_code in lang_map:
        return lang_map[speech_lang_code]
    else:
        # Try to extract just the language part (e.g., 'es' from 'es-ES')
        base_lang = speech_lang_code.split('-')[0].lower()
        return base_lang if base_lang else "en"


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
            
            # Record audio from microphone, using user's preferred language if specified
            text, lang = record_audio(expected_language=user_language)
            
            if text:
                # Store the detected language for later use
                detected_language = lang
                print(f"🌐 Language: {lang}")
                print(f"📝 Original text: {text}")
                
                # Translate to English using LibreTranslate
                source_lang = convert_to_libretranslate_code(lang)
                translated_text = translate_text(text, source_lang, "en")
                
                print(f"🇬🇧 English translation: {translated_text}")
                
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
            
            # Record audio from microphone in English
            text, _ = record_audio(expected_language="en-US")
            
            if text:
                print(f"📝 English text: {text}")
                
                # Translate back to the original language using LibreTranslate
                target_lang = convert_to_libretranslate_code(detected_language)
                translated_text = translate_text(text, "en", target_lang)
                
                print(f"🌍 Translation to {detected_language}: {translated_text}")
                
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
    global user_language
    
    print("="*50)
    print("🎙️  ElevenLabs Voice Translation")
    print("="*50)
    
    # Ask user which language they'll speak for better accuracy
    print("\n🌍 Which language will you speak? (for better recognition)")
    print("Options: en-US (English), es-ES (Spanish), fr-FR (French), de-DE (German)")
    print("         it-IT (Italian), pt-BR (Portuguese), hi-IN (Hindi), etc.")
    print("Or press Enter to auto-detect (slower, less accurate)")
    
    user_input = input("\nEnter language code (or press Enter): ").strip()
    
    if user_input:
        user_language = user_input
        print(f"✅ Will listen for {user_input}")
    else:
        user_language = None
        print("✅ Will try to auto-detect language (may take longer)")
    
    print("\n📋 Instructions:")
    print("  • Press 'v' - Record in your language (translates to English)")
    print("  • Press 'o' - Record in English (translates to original language)")
    print("  • Press 'q' - Quit the program")
    print("\n⏳ Waiting for key press...\n")
    
    # Start listening for keyboard events
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    # Check if API key is set
    if not ELEVEN_LABS_API_KEY:
        print("❌ Error: ELEVEN_LABS_API_KEY environment variable not set!")
        print("Please set it with: export ELEVEN_LABS_API_KEY='your-api-key'")
        exit(1)
    
    print("✅ Using LibreTranslate for free translation (no API key needed!)")
    main()
