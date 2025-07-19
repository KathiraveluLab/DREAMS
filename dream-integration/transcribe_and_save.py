import whisper
from pathlib import Path
import sys

def transcribe(audio_path):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        return

    try:
        print("Loading Whisper model...")
        model = whisper.load_model("tiny")

        print("Transcribing audio...")
        result = model.transcribe(str(audio_path))

        output_path = audio_path.with_suffix(".txt")
        
        print("Saving transcript...")
        output_path.write_text(result["text"], encoding="utf-8")

        print(f"Transcription saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred during transcription:\n{e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_and_save.py path/to/audio.mp3")
    else:
        transcribe(sys.argv[1])