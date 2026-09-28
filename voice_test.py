import sounddevice as sd
import soundfile as sf

fs = 16000
seconds = 5

print("🎤 5 seconds pesunga...")

audio = sd.rec(
    int(seconds * fs),
    samplerate=fs,
    channels=1,
    dtype="float32"
)

sd.wait()

sf.write("record.wav", audio, fs)

print("✅ record.wav save aayiduchu")