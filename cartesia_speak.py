"""
Cartesia TTS speak script (cartesia==3.2.0)

Generates reply.wav using your cloned voice and plays it with pygame.

Usage:
    set CARTESIA_API_KEY=sk_car_...
    python cartesia_speak.py
    python cartesia_speak.py "வணக்கம்! எப்படி உதவலாம்?"
"""

import os
import sys
import time

import pygame
from cartesia import Cartesia

# ============================================================
# Configuration
# ============================================================

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
VOICE_ID = "9b55e0f6-351e-43f6-a522-d4ffb0f4df35"
MODEL_ID = "sonic-3"
OUTPUT_FILE = "reply.wav"

DEFAULT_TRANSCRIPT = (
    "வணக்கம்! நான் உங்கள் தமிழ் குரல் உதவியாளர். "
    "எப்படி உதவலாம்?"
)


def generate_speech(client: Cartesia, transcript: str) -> None:
    """Generate speech with Cartesia SDK v3.2.0 and save to reply.wav."""
    response = client.tts.generate(
        model_id=MODEL_ID,
        transcript=transcript,
        voice={
            "mode": "id",
            "id": VOICE_ID,
        },
        output_format={
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 44100,
        },
    )
    response.write_to_file(OUTPUT_FILE)


def play_wav(file_path: str) -> None:
    """Play a WAV file using pygame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    pygame.mixer.init(frequency=44100, size=-16, channels=1)
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.05)

    pygame.mixer.quit()


def main() -> None:
    if not CARTESIA_API_KEY:
        print("[ERROR] Set the CARTESIA_API_KEY environment variable.")
        sys.exit(1)

    transcript = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_TRANSCRIPT

    client = Cartesia(api_key=CARTESIA_API_KEY)

    print("[SYSTEM] Generating speech with Cartesia...")
    print(f"[TEXT] {transcript}")

    generate_speech(client, transcript)

    print(f"[SYSTEM] Saved: {OUTPUT_FILE}")

    print("[SYSTEM] Playing audio...")
    play_wav(OUTPUT_FILE)

    print("[SYSTEM] Done.")


if __name__ == "__main__":
    main()
