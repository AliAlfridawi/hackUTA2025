# 🗣️ Pi Universal Translator (HackUTA 2025)

> **A handheld, AI-powered "Walkie-Talkie" that breaks down language barriers in real-time.**

Built for **HackUTA 2025**, this project turns a Raspberry Pi into a standalone universal translator. By holding a physical button, users can speak in any language; the device transcribes the audio, translates it, and plays back the translation in a natural AI voice. It also features a "conversation mode" that intelligently handles replies.

## 🚀 Features

* **Physical Interface:** Operates like a walkie-talkie with Push-to-Talk (PTT) buttons and LED status indicators.
* **Smart Language Detection:** Automatically identifies the speaker's language.
* **Bi-Directional Conversation:**
    * **Foreign Input:** Translates foreign speech into English.
    * **Contextual Reply:** If you reply in English immediately after hearing a foreign language, the device automatically translates your reply *back* to that foreign language.
* **High-Quality Audio:** Uses **ElevenLabs** for industry-leading Speech-to-Text (Scribe) and Text-to-Speech (TTS).
* **Hardware Accelerated:** Optimized for Raspberry Pi using `gpiozero` and USB audio peripherals.

## 🛠️ Hardware Requirements

To build this device, you will need:

1.  **Raspberry Pi** (3B, 4, or 5 recommended)
2.  **USB Microphone**
3.  **Speaker** (USB or 3.5mm Aux)
4.  **Push Button** (Momentary switch)
5.  **LEDs** (Red and Green) + Resistors (220Ω or 330Ω)
6.  Breadboard and jumper wires

### Pinout Configuration (BCM)

The default wiring configuration defined in `pi_translator_gpio.py`:

| Component | GPIO Pin (BCM) | Function |
| :--- | :--- | :--- |
| **Button 1** | **GPIO 17** | Push-to-Talk (Hold to record) |
| **Red LED** | **GPIO 23** | Recording Indicator |
| **Green LED** | **GPIO 24** | Processing / Ready Indicator |
| **Button 2** | *(Optional)* | Back-Translation Shortcut |

*Note: Pins can be customized in the `.env` file.*

## 💻 Software Setup

### 1. System Dependencies

You need Python 3 and the PortAudio library for audio device management.

```bash
sudo apt-get update
sudo apt-get install python3-pip portaudio19-dev
```
### 2. Install Python Libraries
Install the required dependencies listed in the project:

```pip install numpy sounddevice soundfile gpiozero elevenlabs deep-translator python-dotenv tqdm```
### 3. Environment Configuration
Create a .env file in the root directory to store your API keys and settings.
```
touch .env
nano .env
```
Paste the following configuration into the file:

```
# API Keys (Required)
ELEVEN_API="your_elevenlabs_api_key_here"

# GPIO Configuration (BCM Numbers)
BTN_PTT_GPIO=17
LED_RED_GPIO=23
LED_GREEN_GPIO=24
# BTN_BACK_GPIO=27  # Uncomment to use a second button
```
## 🎮 How to Use
### 1. Run the script:

```
python3 pi_translator_gpio.py
```
### 2. Wait for initialization: The console will verify the audio devices and the Green LED will light up (if configured).

```
To Translate:

Hold the main button (GPIO 17). The Red LED will turn on.

Speak your sentence clearly into the microphone.

Release the button. The Green LED will turn on while the device processes.

The device will speak the translation out loud.

Conversation Mode:

If the device detects you spoke Spanish (e.g., "Hola, cómo estás?"), it translates to English.

If you immediately hold the button again and speak English (e.g., "I am good, thanks"), the device is smart enough to translate that back into Spanish for the other person.
 ```

## 🧩 Technologies Used
Python: Core logic and hardware control.

[ELEVENLABS]:

```Scribe v1 for high-accuracy Speech-to-Text.

Multilingual v2 for realistic Text-to-Speech synthesis.
```

Deep Translator: Wraps Google Translate for text translation logic.

GPIO Zero: Interface for Raspberry Pi GPIO pins (Buttons/LEDs).

SoundDevice: Low-latency audio recording and playback.

Deep Translator: Wraps Google Translate for text translation logic.

GPIO Zero: Interface for Raspberry Pi GPIO pins (Buttons/LEDs).

SoundDevice: Low-latency audio recording and playback.
