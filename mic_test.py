import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("🎤 Speak now...")
    r.adjust_for_ambient_noise(source, duration=1)
    audio = r.listen(source)

print("Recognizing...")

try:
    text = r.recognize_google(audio, language="ta-IN")
    print("You said:", text)

except Exception as e:
    print("Error:", e)