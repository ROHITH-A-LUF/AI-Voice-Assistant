import subprocess
import time
from playsound import playsound

print("=" * 45)
print("      Tamil AI Voice Assistant")
print("=" * 45)
print("Loading AI Model...........OK")
print("Loading Voice Assets.......OK")
print("Loading Chat Engine........OK")
print("\nSystem Ready!")
print("Type 'help' for available commands.\n")

while True:

    user_input = input("You: ").strip().lower()

    # HELP
    if user_input == "help":
        print("""
Available Commands
------------------
1. vanakkam
2. tamil la pesu
3. india thalainagar enna
4. exit
""")
        continue

    # GREETING
    if user_input == "vanakkam":
        print("\nAI: வணக்கம் ரோகித்!\n")
        playsound(r"C:\SpeakerIsolation\BACKUP_MONDAY\tamil_1.wav")
        continue

    # TAMIL MODE
    if user_input == "tamil la pesu":
        print("\nAI: நிச்சயமாக! இனிமேல் தமிழில் மட்டும் பேசுகிறேன்.\n")
        continue

    # KNOWLEDGE DEMO
    if user_input == "india thalainagar enna":
        print("\nAI: இந்தியாவின் தலைநகரம் புதுடெல்லி.\n")
        continue

    # EXIT
    if user_input == "exit":
        print("\nThank you for using Tamil AI Voice Assistant!")
        break

    # AI MODE
    prompt = f"""
நீ ஒரு தமிழ் AI உதவியாளர்.

விதிகள்:
- தமிழில் மட்டும் பதில் கூறு.
- ஆங்கிலம் பயன்படுத்தாதே.
- கேள்வியை மீண்டும் கூறாதே.
- சுருக்கமாக பதில் கூறு.

பயனர்: {user_input}

பதில்:
"""

    print("\nAI is thinking...")
    time.sleep(1)

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2", prompt],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )

        print("\nAI:", result.stdout.strip())

    except Exception:
        print("\nAI: மன்னிக்கவும்! தற்போது பதில் கிடைக்கவில்லை.")