# Voice Translator (Local) — Spanish <-> English (multi-language support)

This project records microphone input, uses ElevenLabs Speech-to-Text to transcribe and detect the spoken language, translates the detected text to English, and uses ElevenLabs Text-to-Speech to speak the English output. The `E` hotkey records a short English reply from a second speaker, transcribes that reply, translates it back into the language detected from the first speaker, and speaks the translated reply.

## Quick setup (Windows PowerShell)
1. Create and activate a virtual environment (recommend Python 3.11):

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements\requirements.txt
```

3. Add your ElevenLabs API key to a `.env` file in the project root:

```
ELEVEN_API=your_api_key_here
# Optional: set VOICE_ID to choose a different voice
# VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

4. Run the program:

```powershell
python main.py
```

## Controls
- Press `R` (release) — start/stop the first speaker recording.
- After the program transcribes and translates the first speaker it will play the English audio and print a message:
  `Ready for back-translation — press 'E' to translate English back to the original language.`
- Press `E` (release) — the program will record a short English reply (default 5s), transcribe it, translate it into the previously-detected language, synthesize that translation, and play it.
- Press `ESC` to exit the program.

## Configuration
- `BACK_RECORD_SECONDS` in `main.py` controls how many seconds the program records for the `E` workflow (default: 5).
- To choose a different voice, set `VOICE_ID` in your `.env` file.

## Notes & Troubleshooting
- Keyboard hotkeys on Windows may require the program to run with administrator permissions to capture global key events. If `R` or `E` doesn't respond, try starting PowerShell as Administrator.
- Microphone access: ensure Windows privacy settings allow the terminal/PowerShell app to access the microphone.
- If `sounddevice` raises an error about devices, run:

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

- If you get strange TTS output or `pcm_16000` errors, check your ElevenLabs API key permissions and the SDK's allowed output formats.

## Extending the project
- Replace the fixed reply recording with voice activity detection (VAD) to record until the speaker stops.
- Swap `deep-translator` for another translation API (e.g., Google Cloud Translation, Azure Translator) for higher-quality translations.

## Files
- `main.py` — main program
- `requirements/requirements.txt` — Python dependencies
- `.env` — store `ELEVEN_API` (not checked into source control)

If you want, I can also add a small `scripts/` folder with convenience PowerShell scripts (create venv, install deps, run) or add a quick status display to `main.py`.
