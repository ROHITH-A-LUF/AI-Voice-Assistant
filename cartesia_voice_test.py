from cartesia import Cartesia

try:
    client = Cartesia(api_key="sk_car_3Ltbe7m1UVvrPvKn5BvvHC")

    print("✅ Connected Successfully!")

    voices = client.voices.list()

    found = False

    for voice in voices:
        if "Mentor_Appa" in voice.name:
            print("\n🎉 Voice Found!")
            print("Name :", voice.name)
            print("ID   :", voice.id)
            found = True

    if not found:
        print("❌ Mentor_Appa voice not found.")

except Exception as e:
    print("ERROR:")
    print(e)