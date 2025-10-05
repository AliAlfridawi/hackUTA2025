# HackUTA2025 Project

This project contains various scripts and tools for audio processing, translation, and AI integration, including:
- Speech recognition
- Text-to-speech
- Language translation
- Integration with ElevenLabs, Google Generative AI, DeepL, and OpenAI

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AliAlfridawi/hackUTA2025.git
   cd hackUTA2025/hackuta
   ```

2. **Create and activate a virtual environment (recommended):**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Install system dependencies (for audio):**
   - On macOS:
     ```sh
     brew install portaudio
     ```

5. **Set up API keys as environment variables as needed.**

## Main Dependencies
- faster-whisper
- soundfile
- speechrecognition
- pyaudio
- pynput
- pygame
- requests
- google-generativeai
- deepl
- openai
- elevenlabs
- googletrans

## Usage
- See individual script files for usage instructions and entry points.
- Example:
  ```sh
  python hackuta/main.py
  ```

## License
MIT
