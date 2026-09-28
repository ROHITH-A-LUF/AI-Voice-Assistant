import speech_recognition as sr

r = sr.Recognizer()

with sr.AudioFile("record.wav") as source:
    audio = r.record(source)

try:
    text = r.recognize_google(audio, language="ta-IN")
    print("நீங்கள் சொன்னது :", text)

except Exception as e:
    print("Error:", e)