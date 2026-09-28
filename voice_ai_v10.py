# ============================================================
# Voice AI V10
# Whisper STT → Routing Layer → Ollama / Guard → Cartesia
# ============================================================

import os
import re
import string
import subprocess
import tempfile
import time
import logging
from logging.handlers import RotatingFileHandler

import pygame
import requests
import sounddevice as sd
import soundfile as sf
from cartesia import Cartesia
from faster_whisper import WhisperModel

# ============================================================
# Configuration
# ============================================================

# API Keys and Models
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
VOICE_ID = "9b55e0f6-351e-43f6-a522-d4ffb0f4df35"
MODEL_ID = "sonic-3"
OUTPUT_FILE = "reply.wav"

OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TIMEOUT = (3.05, 27.0)  # (connect timeout, read timeout)

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5

# Chat Logic
EXIT_COMMANDS = ["வெளியேறு", "நிறுத்து", "exit", "bye"]

# Whisper initial prompt
WHISPER_INITIAL_PROMPT = (
    "வணக்கம். நான் உங்கள் தமிழ் குரல் உதவியாளர். நீங்கள் தமிழில் பேசலாம். "
    "பொதுவாக தமிழ்நாடு, இந்தியா, தலைநகர், நகைச்சுவை, செயற்கை நுண்ணறிவு "
    "போன்ற கேள்விகளுக்கு நான் பதில் அளிக்கிறேன்."
)

# Logging Configuration
LOG_FILE = "voice_ai_v10.log"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3

# Setup Logger
logger = logging.getLogger("VoiceAI")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

conversation_history = [
    {
        "role": "system",
        "content": (
            "நீ ஒரு தமிழ் AI உதவியாளர். விதிகள்:\n"
            "- வினாக்களுக்குத் துல்லியமான மற்றும் உண்மையான தகவல்களை மட்டுமே தமிழில் சுருக்கமாக விடையளிக்கவும்.\n"
            "- புவியியல், வரலாறு, அறிவியல் அல்லது நடப்பு நிகழ்வுகள் பற்றி உனக்குத் தெரியாத தகவலை கற்பனையாகவோ தவறாகவோ கூறக்கூடாது.\n"
            "- விடை தெரியவில்லை என்றாலோ அல்லது உறுதியற்ற தகவலாக இருந்தாலோ கண்டிப்பாக 'எனக்குத் தெரியாது.' என்று மட்டுமே பதில் கூற வேண்டும்."
        )
    }
]

whisper_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
)

cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

# ============================================================
# Centralized Personnel (Time-Sensitive Constants)
# ============================================================
CURRENT_PM_INDIA = "திரு. நரேந்திர மோடி"
CURRENT_PRESIDENT_INDIA = "திருமதி. திரௌபதி முர்மு"
CURRENT_CM_TAMILNADU = "திரு. மு. க. ஸ்டாலின்"

# ============================================================
# Centralized Keyword Constants and Mappings
# ============================================================

# Organized Whisper Spelling Variations Normalization Map
GREETINGS_NORM = {
    "காலை வழக்க்கம்": "காலை வணக்கம்",
    "காலை வணககம்": "காலை வணக்கம்",
    "மாலை வழக்க்கம்": "மாலை வணக்கம்"
}

IDENTITY_NORM = {
    "பெயரென்ன": "பெயர் என்ன",
    "பேரென்ன": "பெயர் என்ன",
    "உருவாக்கியது": "உருவாக்கினார்"
}

GEOGRAPHY_NORM = {
    "தலைநகார்": "தலைநகர்",
    "செனை": "சென்னை"
}

SYMBOLS_NORM = {
    "டேசியே": "தேசிய",
    "தேச்சிய": "தேசிய",
    "தேச்சியை": "தேசிய",
    "விலங்கி": "விலங்கு",
    "கीதமென்ன": "கீதம்",
    "கீதமென்ன": "கீதம்",
    "தேசியகீதமென்ன": "தேசிய கீதம்"
}

MATH_NORM = {
    "குட்டல்": "கூட்டல்",
    "வகுத்தலி": "வகுத்தல்",
    "மூண்டிரு": "மூன்று",
    "இரண்டியவழவு": "இரண்டு",
    "வகுத்தில்": "வகுத்தல்"
}

EXIT_NORM = {
    "வெளியிரு": "வெளியேறு",
    "விளிரெய்": "வெளியேறு"
}

TECH_NORM = {
    "ஏ ஐ": "AI",
    "ஏஐ": "AI",
    "செயற்கைநுண்ணறிவு": "செயற்கை நுண்ணறிவு",
    "பைதான்": "Python",
    "சி பிளஸ்": "C++",
    "ஜாவா": "Java"
}

GENERAL_NORM = {
    "தமில்லாடு": "தமிழ்நாடு",
    "தமில்லாட்டின்": "தமிழ்நாட்டின்",
    "நாய் சுவை": "நகைச்சுவை",
    "நாய்சுவை": "நகைச்சுவை",
    "கிருகம்": "கிரகம்"
}

# Merge all normalized categories
NORMALIZATION_MAP = {}
NORMALIZATION_MAP.update(GREETINGS_NORM)
NORMALIZATION_MAP.update(IDENTITY_NORM)
NORMALIZATION_MAP.update(GEOGRAPHY_NORM)
NORMALIZATION_MAP.update(SYMBOLS_NORM)
NORMALIZATION_MAP.update(MATH_NORM)
NORMALIZATION_MAP.update(EXIT_NORM)
NORMALIZATION_MAP.update(TECH_NORM)
NORMALIZATION_MAP.update(GENERAL_NORM)

PHRASE_NORMALIZATION_MAP = {
    # Exit Commands
    "விளி ரெய்": "வெளியேறு",
    # Greetings variations
    "காலை வழக்க்கம்": "காலை வணக்கம்",
    "காலை வணககம்": "காலை வணக்கம்",
    "மாலை வழக்க்கம்": "மாலை வணக்கம்"
}

# Entity Synonyms / Aliases
ENTITY_ALIASES = {
    "தமிழ்நாடு": ["தமிழ்நாடு", "தமிழ் நாடு", "தமிழ்நாட்டின்", "தமிழ் நாட்டின்", "தமிழ்நாட்டை", "தமிழ்நாட்டிற்கு"],
    "இந்தியா": ["இந்தியா", "இந்தியாவின்", "இந்தியாவை", "இந்தியாவிற்கு"],
    "AI": ["ai", "artificial intelligence", "ஏ ஐ", "ஏஐ", "செயற்கை நுண்ணறிவு", "செயற்கைநுண்ணறிவு", "செயற்கை நுண்ணறிவின்"],
    "Python": ["python", "பைதான்", "பைத்தான்", "பைதானின்"],
    "C++": ["c++", "சி பிளஸ்", "சி பிளஸ் பிளஸ்", "சி++"],
    "Java": ["java", "ஜாவா", "ஜாவாவின்"],
    "Computer Science": ["computer science", "கணினி அறிவியல்", "கணினிஅறிவியல்", "கணினி அறிவியல் பாடம்"],
    "C": ["c programming", "c language", "சி நிரலாக்க மொழி", "சி மொழி", "சி லேங்குவேஜ்"],
    "JavaScript": ["javascript", "js", "ஜாவாஸ்கிரிப்ட்", "ஜாவா ஸ்கிரிப்ட்"],
    "Machine Learning": ["machine learning", "ml", "மெஷின் லேர்னிங்", "இயந்திரக் கற்றல்", "இயந்திர கற்றல்"],
    "Deep Learning": ["deep learning", "dl", "டீப் லேர்னிங்", "ஆழ்ந்த கற்றல்", "ஆழ் கற்றல்"],
    "Neural Network": ["neural network", "நியூரல் நெட்வொர்க்", "நரம்பியல் வலைப்பின்னல்", "நரம்பியல் நெட்வொர்க்"],
    "LLM": ["llm", "லார்ஜ் லேங்குவேஜ் மாடல்", "பெரிய மொழி மாதிரி", "பெரிய மொழி மாதிரிகள்"],
    "Algorithm": ["algorithm", "அல்காரிதம்", "நெறிமுறை", "வழிமுறை"],
    "Data Structure": ["data structure", "டேட்டா ஸ்ட்ரக்சர்", "தரவுக் கட்டமைப்பு", "தரவு கட்டமைப்பு"],
    "Operating System": ["operating system", "ஓஎஸ்", "இயக்க முறைமை", "இயக்கமுறைமை", "os"],
    "Database": ["database", "டேட்டாபேஸ்", "தரவுத்தளம்", "தரவு தளம்"],
    "Computer Network": ["computer network", "கம்ப்யூட்டர் நெட்வொர்க்", "கணினி வலைப்பின்னல்", "கணினி நெட்வொர்க்"]
}

# Topic Synonyms / Aliases
TOPIC_ALIASES = {
    "தலைநகர்": ["தலைநகர்", "தலைநகரம்", "தலை நகர்", "தலை நகரம்"],
    "தேசிய கீதம்": ["தேசிய கீதம்", "தேசியகீதம்"],
    "தேசிய விலங்கு": ["தேசிய விலங்கு", "தேசியவிலங்கு"],
    "தேசிய மலர்": ["தேசிய மலர்", "தேசியமலர்"],
    "தேசிய பறவை": ["தேசிய பறவை", "தேசியப்பறவை"],
    "முதலமைச்சர்": ["முதலமைச்சர்", "முதல் அமைச்சர்", "முதல்வர்", "மு.க.ஸ்டாலின்", "ஸ்டாலின்"],
    "குடியரசுத் தலைவர்": ["குடியரசுத் தலைவர்", "குடியரசு தலைவர்", "ஜனாதிபதி"],
    "பிரதமர்": ["பிரதமர்", "பிரதம மந்திரி", "நரேந்திர மோடி", "மோடி"],
    "விளக்கம்": ["என்ன", "என்றால் என்ன", "விளக்கம்", "விளக்கு", "பற்றி", "அறிமுகம்"],
    "தேசிய கொடி": ["தேசிய கொடி", "தேசியக்கொடி", "தேசிய கொடியின்", "தேசியக் கொடி"],
    "தேசிய பாடல்": ["தேசிய பாடல்", "தேசியப்பாடல்", "வந்தே மாதரம்"],
    "தேசிய கனி": ["தேசிய கனி", "தேசியக் கனி", "தேசிய பழம்"],
    "தேசிய மரம்": ["தேசிய மரம்", "தேசியமரம்", "தேசிய மரம் எது"],
    "தேசிய நதி": ["தேசிய நதி", "தேசிய நதி எது", "தேசிய ஆறு"],
    "அதிகாரப்பூர்வ மொழி": ["அதிகாரப்பூர்வ மொழி", "அதிகாரபூர்வ மொழி", "மொழி", "தமிழ்நாடு மொழி"],
    "மாநில விலங்கு": ["மாநில விலங்கு", "மாநிலவிலங்கு", "மாநில விலங்கு எது"],
    "மாநில பறவை": ["மாநில பறவை", "மாநிலப்பறவை", "மாநில பறவை எது"],
    "மாநில மலர்": ["மாநில மலர்", "மாநிலமலர்", "மாநில பூ"],
    "மாநில மரம்": ["மாநில மரம்", "மாநிலமரம்", "மாநில மரம் எது"],
    "மாநில நடனம்": ["மாநில நடனம்", "மாநில நடனம் எது", "நடனம்"],
    "மாநில பாடல்": ["மாநில பாடல்", "தமிழ்த்தாய் வாழ்த்து", "மாநிலப்பாடல்"],
    "மாநில கனி": ["மாநில கனி", "மாநிலக் கனி", "மாநில பழம்"],
    "படைப்பாளர்": ["படைப்பாளர்", "உருவாக்கியவர்", "கண்டுபிடித்தவர்", "எழுதியவர்"]
}

# Identity Synonyms / Aliases
IDENTITY_ALIASES = {
    "உன் பெயர்": ["உன் பெயர்", "உன் பேர்", "உன்னுடைய பெயர்", "உன்னுடைய பேர்", "பெயர் என்ன", "பேர் என்ன"],
    "நீ யார்": ["நீ யார்", "யார் நீ", "உன்னை பற்றி", "உன்னைப் பற்றி"],
    "நீ என்ன செய்ய முடியும்": ["என்ன செய்ய முடியும்", "என்ன செய்வாய்", "நீ என்ன செய்ய", "உன்னால் என்ன செய்ய"],
    "உன்னை யார் உருவாக்கினார்": ["யார் உருவாக்கினார்", "யார் உருவாக்கியது", "உன்னை உருவாக்கியது யார்", "எழுதியது யார்"]
}

# Exit Commands aliases
EXIT_ALIASES = {
    "வெளியேறு": ["வெளியேறு", "வெளியிரு", "பெளியரி", "வெளிரி", "வெளியேறுங்கள்", "வெளியேறுங்க", "exit", "bye"],
    "நிறுத்து": ["நிறுத்து", "நிறுத்துங்கள்", "நிறுத்துங்க", "நிறுது", "நிருத்து", "நிருது"]
}

# Math aliases
MATH_ALIASES = {
    "கூட்டல்": ["கூட்டல்", "பிளஸ்"],
    "கழித்தல்": ["கழித்தல்", "மைனஸ்"],
    "பெருக்கல்": ["பெருக்கல்"],
    "வகுத்தல்": ["வகுத்தல்"],
    "பூஜ்ஜியம்": ["பூஜ்ஜியம்"],
    "ஒன்று": ["ஒன்று"],
    "இரண்டு": ["இரண்டு"],
    "மூன்று": ["மூன்று"],
    "நான்கு": ["நான்கு"],
    "ஐந்து": ["ஐந்து"],
    "ஆறு": ["ஆறு"],
    "ஏழு": ["ஏழு"],
    "எட்டு": ["எட்டு"],
    "ஒன்பது": ["ஒன்பது"],
    "பத்து": ["பத்து"]
}

# Common phrase targets for entity/topic matching
PHRASE_ALIASES = {
    "தேசிய விலங்கு": ["தேசிய விலங்கு", "தேசியவிலங்கு", "தேசிய விலங்கி", "தேசியவிலங்கி", "தேசிய விலங்கின்"],
    "தேசிய கீதம்": ["தேசிய கீதம்", "தேசியகீதம்", "தேசியகீதமென்ன", "தேசிய கீதமென்ன"],
    "தேசிய பறவை": ["தேசிய பறவை", "தேசியப்பறவை"],
    "தேசிய மலர்": ["தேசிய மலர்", "தேசியமலர்"],
    "தமிழ்நாடு தலைநகர்": ["தமிழ்நாடு தலைநகர்", "தமிழ் நாடு தலைநகர்", "தமிழ்நாட்டின் தலைநகர்", "தமிழ்நாட்டின் தலைநகரம்", "தமிழ் நாட்டின் தலைநகர்", "தமிழ்நாட்டின் தலை நகர்"],
    "இந்தியா தலைநகர்": ["இந்தியா தலைநகர்", "இந்தியாவின் தலைநகர்", "இந்தியாவின் தலைநகரம்"]
}

# Greetings Keywords
GREETINGS_KEYWORDS = ["வணக்கம்", "ஹலோ", "காலை வணக்கம்", "மாலை வணக்கம்", "hello", "hi"]

# Tamil Numbers Map
TAMIL_NUMBERS_MAP = {
    "பூஜ்ஜியம்": "0",
    "ஒன்று": "1",
    "இரண்டு": "2",
    "மூன்று": "3",
    "நான்கு": "4",
    "ஐந்து": "5",
    "ஆறு": "6",
    "ஏழு": "7",
    "எட்டு": "8",
    "ஒன்பது": "9",
    "பத்து": "10"
}

# Entity-Topic Database mapped to canonical keys
ENTITY_TOPIC_DB = {
    ("தமிழ்நாடு", "தலைநகர்"): "தமிழ்நாட்டின் தலைநகரம் சென்னை ஆகும்.",
    ("இந்தியா", "தலைநகர்"): "இந்தியாவின் தலைநகரம் புது தில்லி ஆகும்.",
    ("AI", "விளக்கம்"): "செயற்கை நுண்ணறிவு (Artificial Intelligence அல்லது AI) என்பது கணினிகள் அல்லது இயந்திரங்கள் மனிதர்களைப் போல சிந்தித்து செயல்பட உருவாக்கப்பட்ட ஒரு தொழில்நுட்பம் ஆகும்.",
    ("இந்தியா", "தேசிய கீதம்"): "இந்தியாவின் தேசிய கீதம் 'ஜன கண மன' ஆகும். இதனை இரவீந்திரநாத் தாகூர் இயற்றினார்.",
    ("இந்தியா", "தேசிய விலங்கு"): "இந்தியாவின் தேசிய விலங்கு வங்காளப் புலி ஆகும்.",
    ("இந்தியா", "தேசிய மலர்"): "இந்தியாவின் தேசிய மலர் தாமரை மலர் ஆகும்.",
    ("இந்தியா", "தேசிய பறவை"): "இந்தியாவின் தேசிய பறவை மயில் ஆகும்.",
    ("தமிழ்நாடு", "முதலமைச்சர்"): f"தற்போது தமிழ்நாட்டின் முதலமைச்சர் {CURRENT_CM_TAMILNADU} ஆவார்.",
    ("இந்தியா", "குடியரசுத் தலைவர்"): f"தற்போது இந்தியாவின் குடியரசுத் தலைவர் {CURRENT_PRESIDENT_INDIA} ஆவார்.",
    ("இந்தியா", "பிரதமர்"): f"தற்போது இந்தியாவின் பிரதமர் {CURRENT_PM_INDIA} ஆவார்.",
    ("Python", "விளக்கம்"): "Python என்பது ஒரு எளிய மற்றும் மிகவும் பிரபலமான கணினி நிரலாக்க மொழியாகும் (Programming Language). இது எளிதாகக் கற்கக்கூடியது.",
    ("C++", "விளக்கம்"): "C++ என்பது ஒரு சக்திவாய்ந்த மற்றும் பல்துறை கணினி நிரலாக்க மொழியாகும். இது செயல்திறன் மற்றும் மென்பொருள் உருவாக்கத்தில் பரவலாகப் பயன்படுத்தப்படுகிறது.",
    ("Java", "விளக்கம்"): "Java என்பது ஒரு இயங்குதளத்தை சாராத (Platform-Independent) மற்றும் பொருளை அடிப்படையாகக் கொண்ட (Object-Oriented) நிரலாக்க மொழியாகும்.",
    ("Computer Science", "விளக்கம்"): "கணினி அறிவியல் (Computer Science) என்பது கணினிகள், கணினி மென்பொருள் மற்றும் கணக்கீட்டு அமைப்புகளைப் பற்றிய ஒரு படிப்பாகும்.",
    ("இந்தியா", "தேசிய கொடி"): "இந்தியாவின் தேசியக் கொடி மூவர்ணக் கொடி ஆகும். இதில் காவி, வெள்ளை, பச்சை நிறங்களும், நடுவில் அசோக சக்கரமும் உள்ளன.",
    ("இந்தியா", "தேசிய பாடல்"): "இந்தியாவின் தேசியப் பாடல் 'வந்தே மாதரம்' ஆகும். இதனை பங்கிம் சந்திர சாட்டர்ஜி இயற்றினார்.",
    ("இந்தியா", "தேசிய கனி"): "இந்தியாவின் தேசியக் கனி மாம்பழம் ஆகும்.",
    ("இந்தியா", "தேசிய மரம்"): "இந்தியாவின் தேசிய மரம் ஆலமரம் ஆகும்.",
    ("இந்தியா", "தேசிய நதி"): "இந்தியாவின் தேசிய நதி கங்கை நதி ஆகும்.",
    ("தமிழ்நாடு", "அதிகாரப்பூர்வ மொழி"): "தமிழ்நாட்டின் அதிகாரப்பூர்வ மொழி தமிழ் ஆகும்.",
    ("தமிழ்நாடு", "மாநில விலங்கு"): "தமிழ்நாட்டின் மாநில விலங்கு வரையாடு ஆகும்.",
    ("தமிழ்நாடு", "மாநில பறவை"): "தமிழ்நாட்டின் மாநிலப் பறவை மரகதப்புறா ஆகும்.",
    ("தமிழ்நாடு", "மாநில மலர்"): "தமிழ்நாட்டின் மாநில மலர் செங்காந்தள் மலர் ஆகும்.",
    ("தமிழ்நாடு", "மாநில மரம்"): "தமிழ்நாட்டின் மாநில மரம் பனை மரம் ஆகும்.",
    ("தமிழ்நாடு", "மாநில நடனம்"): "தமிழ்நாட்டின் மாநில நடனம் பரதநாட்டியம் ஆகும்.",
    ("தமிழ்நாடு", "மாநில பாடல்"): "தமிழ்நாட்டின் மாநிலப் பாடல் 'நீராரும் கடலுடுத்த' எனத் தொடங்கும் தமிழ்த்தாய் வாழ்த்து ஆகும்.",
    ("தமிழ்நாடு", "மாநில கனி"): "தமிழ்நாட்டின் மாநிலக் கனி பலாப்பழம் ஆகும்.",
    ("C", "விளக்கம்"): "C என்பது ஒரு அடிப்படை மற்றும் மிகவும் பிரபலமான கணினி நிரலாக்க மொழியாகும். இது கணினி இயக்க முறைமைகளை (Operating Systems) உருவாக்கப் பயன்படுத்தப்படுகிறது.",
    ("JavaScript", "விளக்கம்"): "JavaScript என்பது இணையதளங்களை (Websites) உருவாக்கப் பயன்படும் ஒரு ஸ்கிரிப்டிங் மொழியாகும். இது இணையப் பக்கங்களை ஊடாடத்தக்கதாக மாற்றுகிறது.",
    ("Machine Learning", "விளக்கம்"): "இயந்திரக் கற்றல் (Machine Learning அல்லது ML) என்பது கணினிகள் தரவுகளிலிருந்து தாங்களாகவே கற்றுக்கொண்டு முன்னறிவிப்புகளைச் செய்ய உதவும் AI-ன் ஒரு பிரிவு ஆகும்.",
    ("Deep Learning", "விளக்கம்"): "ஆழ்ந்த கற்றல் (Deep Learning அல்லது DL) என்பது செயற்கை நரம்பியல் வலைப்பின்னல்களைப் பயன்படுத்தி சிக்கலான தரவுகளை பகுப்பாய்வு செய்யும் ஒரு தொழில்நுட்பம் ஆகும்.",
    ("Neural Network", "விளக்கம்"): "நரம்பியல் வலைப்பின்னல் (Neural Network) என்பது கணினிகள் தரவுகளை மனித மூளையைப் போல பகுப்பாய்வு செய்ய வடிவமைக்கப்பட்ட செயற்கை நரம்புகளின் கட்டமைப்பு ஆகும்.",
    ("LLM", "விளக்கம்"): "பெரிய மொழி மாதிரி (Large Language Model அல்லது LLM) என்பது மனித மொழியைப் புரிந்து கொண்டு உரையாடல்களை உருவாக்கப் பயன்படும் ஒரு மேம்பட்ட AI மாதிரி ஆகும்.",
    ("Algorithm", "விளக்கம்"): "நெறிமுறை (Algorithm) என்பது ஒரு குறிப்பிட்ட பணியைச் செய்ய அல்லது ஒரு சிக்கலைத் தீர்க்கப் பயன்படுத்தப்படும் படிப்படியான வழிமுறைகளின் தொகுப்பு ஆகும்.",
    ("Data Structure", "விளக்கம்"): "தரவுக் கட்டமைப்பு (Data Structure) என்பது கணினியில் தரவுகளை திறமையாக சேமிக்கவும் ஒழுங்கமைக்கவும் பயன்படுத்தப்படும் ஒரு குறிப்பிட்ட வடிவம் ஆகும் (உதாரணமாக Array, Stack).",
    ("Operating System", "விளக்கம்"): "இயக்க முறைமை (Operating System அல்லது OS) என்பது பயனர் மற்றும் கணினி வன்பொருளுக்கு இடையே ஒரு பாலமாகச் செயல்படும் முக்கிய மென்பொருள் ஆகும் (உதாரணமாக Windows, Linux).",
    ("Database", "விளக்கம்"): "தரவுத்தளம் (Database) என்பது கணினி அமைப்பில் சேமிக்கப்பட்டு முறையாக நிர்வகிக்கப்படும் தரவுகளின் ஒழுங்கமைக்கப்பட்ட தொகுப்பு ஆகும்.",
    ("Computer Network", "விளக்கம்"): "கணினி வலைப்பின்னல் (Computer Network) என்பது தகவல் மற்றும் வளங்களை பகிர்ந்து கொள்வதற்காக ஒன்றோடொன்று இணைக்கப்பட்ட கணினிகளின் தொகுப்பு ஆகும்.",
    ("Python", "படைப்பாளர்"): "Python நிரலாக்க மொழியை உருவாக்கியவர் கைடோ வான் ரோசம் (Guido van Rossum) ஆவார்.",
    ("Java", "படைப்பாளர்"): "Java நிரலாக்க மொழியை உருவாக்கியவர் ஜேம்ஸ் கோஸ்லிங் (James Gosling) ஆவார்.",
    ("C++", "படைப்பாளர்"): "C++ நிரலாக்க மொழியை உருவாக்கியவர் ஜார்னே ஸ்ட்ரூஸ்ட்ரப் (Bjarne Stroustrup) ஆவார்."
}

# ============================================================
# Record Voice
# ============================================================


def record_audio(duration=RECORD_SECONDS):
    """
    Record audio from the microphone.
    Returns the temporary WAV file path.
    """
    logger.info("Starting audio recording...")
    print("\n[SYSTEM] Listening...")

    try:
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        )
        sd.wait()
    except Exception as e:
        logger.error("Microphone recording failed", exc_info=True)
        print(f"[ERROR] Microphone recording failed: {e}")
        return None

    # Normalize volume if recording is too quiet to improve Whisper recognition
    import numpy as np
    max_val = np.max(np.abs(recording))
    if 1000 < max_val < 15000:
        scale_factor = 31128 / max_val
        recording = np.clip(recording * scale_factor, -32768, 32767).astype(np.int16)

    # Windows temporary file handling: close the file handle immediately
    # to release the write-lock before soundfile writes to it.
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        sf.write(
            temp_file_path,
            recording,
            SAMPLE_RATE,
        )
        logger.info("Audio recording saved successfully.")
        print("[SYSTEM] Recording complete.")
        return temp_file_path
    except Exception as e:
        logger.error("Failed to write audio file", exc_info=True)
        print(f"[ERROR] Failed to save recorded audio: {e}")
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        return None


# ============================================================
# Speech-to-Text
# ============================================================


def normalize_tamil_text(text):
    """
    Normalize Tamil text: Unicode normalization, whitespace cleaning,
    and replacing common ASR mistakes.
    """
    if not text:
        return ""
    import unicodedata
    # Unicode normalize
    text = unicodedata.normalize("NFC", text)
    # Remove extra spaces
    text = " ".join(text.split())

    # Dictionary replacements
    replacements = {
        "தமில்லாடு": "தமிழ்நாடு",
        "தமில்லாட்டின்": "தமிழ்நாட்டின்",
        "தலைநகார்": "தலைநகர்",
        "நாய் சுவை": "நகைச்சுவை",
        "நாய்சுவை": "நகைச்சுவை",
        "ஏ ஐ": "AI",
        "ஏஐ": "AI"
    }

    for target in ["தமில்லாட்டின்", "தமில்லாடு", "தலைநகார்", "நாய் சுவை", "நாய்சுவை", "ஏ ஐ", "ஏஐ"]:
        if target in replacements:
            text = text.replace(target, replacements[target])

    return text


def speech_to_text(audio_path):
    """
    Convert speech to text using Faster-Whisper.
    Deletes the temporary recording after transcription.
    """

    if not audio_path:
        return ""

    try:
        logger.info("Starting Faster-Whisper transcription...")
        print("[SYSTEM] Converting speech to text...")

        segments, info = whisper_model.transcribe(
            audio_path,
            language="ta",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                speech_pad_ms=250,
                min_silence_duration_ms=300
            ),
            initial_prompt=WHISPER_INITIAL_PROMPT,
            condition_on_previous_text=False,
            temperature=0.0,
        )

        text = ""

        for segment in segments:
            text += segment.text

        text = text.strip()
        logger.info("Faster-Whisper transcription completed successfully.")
        print(f"[USER] {text}")

        return text

    except Exception as e:
        logger.error("Speech-to-Text conversion failed", exc_info=True)
        print("[ERROR] Speech-to-Text failed.")
        print(e)
        return ""

    finally:
        # Robust error handling for temporary file removal
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info("Temporary recording file deleted.")
            except Exception as e:
                logger.warning("Failed to remove temporary file", exc_info=True)
                print(f"[WARNING] Failed to remove temporary file: {e}")


# ============================================================
# Intelligence Layer (Classifier, Router, Guard, Validator)
# ============================================================


def levenshtein_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return 0.0
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    distance = previous_row[-1]
    return 1.0 - (distance / len(s1))


def fuzzy_match_word(word, candidates, threshold):
    """
    Fuzzy match a word against a list of candidates.
    Returns the best matching candidate and the similarity score if above threshold.
    Otherwise returns (None, 0.0).
    """
    if not word or len(word) < 3:
        return None, 0.0
        
    best_match = None
    best_score = 0.0
    for cand in candidates:
        score = levenshtein_similarity(word, cand)
        if score >= threshold and score > best_score:
            best_score = score
            best_match = cand
            
    return best_match, best_score


def print_fuzzy_log(category, original, matched, score):
    # Only print logs when an actual correction occurs (original != matched)
    # and similarity is not 1.0
    if original != matched and score < 1.0:
        print("\n[FUZZY MATCH]")
        print(f"Category   : {category}")
        print(f"Original   : {original}")
        print(f"Matched    : {matched}")
        print(f"Similarity : {round(score, 3)}")


def normalize_query(text):
    """
    Standardizes the user query for classifier and routing layers.
    Performs basic unicode normalization, strips spacing and punctuation,
    and maps Whisper misrecognitions via word-aware/phrase-aware maps.
    """
    if not text:
        return ""
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    
    punctuation_chars = '.,!?;:"\'()[]{}<>*-+=_\\/|~`@#$%^&'
    for char in punctuation_chars:
        text = text.replace(char, ' ')
    text = " ".join(text.split())
    
    original_normalized = text
    
    # 1. Phrase-aware matching (multi-word replacements)
    for phrase, replacement in PHRASE_NORMALIZATION_MAP.items():
        padded = f" {text} "
        if f" {phrase} " in padded:
            text = padded.replace(f" {phrase} ", f" {replacement} ").strip()
            
    # 2. Word-aware matching
    words = text.split()
    for i, w in enumerate(words):
        if w in NORMALIZATION_MAP:
            words[i] = NORMALIZATION_MAP[w]
    text = " ".join(words)
    
    # 3. Log normalizer changes if anything changed
    if text != original_normalized:
        print("\n[NORMALIZER]")
        print(f"Original Query   : {original_normalized}")
        print(f"Normalized Query : {text}")
        
    return text


def clean_exit(text):
    words = text.split()
    corrections = []
    exit_candidates = []
    for aliases in EXIT_ALIASES.values():
        exit_candidates.extend(aliases)
        
    for idx, w in enumerate(words):
        if w in GREETINGS_KEYWORDS:
            continue
        if w not in exit_candidates:
            matched_word, score = fuzzy_match_word(w, exit_candidates, 0.75)
            if matched_word:
                corrections.append((w, matched_word, "Exit Keywords", score))
                words[idx] = matched_word
                
    # Map exit command aliases to canonical keys
    for idx, w in enumerate(words):
        for canonical, aliases in EXIT_ALIASES.items():
            if w in aliases:
                words[idx] = canonical
                break
    return " ".join(words), corrections


def clean_math(text):
    words = text.split()
    corrections = []
    math_ops = ["கூட்டல்", "கழித்தல்", "பெருக்கல்", "வகுத்தல்", "பிளஸ்", "மைனஸ்"]
    math_nums = list(TAMIL_NUMBERS_MAP.keys())
    
    for idx, w in enumerate(words):
        if w in GREETINGS_KEYWORDS:
            continue
            
        # 1. Automatic splitting of merged math expression: <Number> + <Operator>
        split_found = False
        for split_idx in range(3, len(w) - 2):
            left = w[:split_idx]
            right = w[split_idx:]
            
            matched_num, score_num = left, 1.0
            if left not in math_nums:
                matched_num, score_num = fuzzy_match_word(left, math_nums, 0.7)
                
            if matched_num:
                matched_op, score_op = right, 1.0
                if right not in math_ops:
                    matched_op, score_op = fuzzy_match_word(right, math_ops, 0.7)
                    
                if matched_op:
                    words[idx] = f"{matched_num} {matched_op}"
                    if score_num < 1.0:
                        corrections.append((left, matched_num, "Math Keywords", score_num))
                    if score_op < 1.0:
                        corrections.append((right, matched_op, "Math Keywords", score_op))
                    split_found = True
                    break
        if split_found:
            continue
            
        # Standalone math keyword fuzzy match
        math_candidates = set(math_ops + math_nums)
        if w not in math_candidates:
            matched, score = fuzzy_match_word(w, math_candidates, 0.7)
            if matched:
                corrections.append((w, matched, "Math Keywords", score))
                words[idx] = matched
                
    return " ".join(words), corrections


def clean_identity(text):
    words = text.split()
    corrections = []
    identity_candidates = set(w for aliases in IDENTITY_ALIASES.values() for a in aliases for w in a.split())
    
    for idx, w in enumerate(words):
        if w in GREETINGS_KEYWORDS:
            continue
        if w not in identity_candidates:
            matched, score = fuzzy_match_word(w, identity_candidates, 0.7)
            if matched:
                corrections.append((w, matched, "Identity Keywords", score))
                words[idx] = matched
                
    return " ".join(words), corrections


def clean_entity_topic(text):
    corrections = []
    
    # Priority 1 & 2: Exact phrase & Alias phrase matching
    for canonical, aliases in PHRASE_ALIASES.items():
        for alias in aliases:
            padded_text = f" {text} "
            padded_alias = f" {alias} "
            if padded_alias in padded_text:
                if alias != canonical:
                    corrections.append((alias, canonical, "Topic Keywords", 1.0))
                text = padded_text.replace(padded_alias, f" {canonical} ").strip()
                break
                
    # Token-aware matching for role names & phrases
    ROLE_PHRASES = [
        "பிரதமர் யார்",
        "முதலமைச்சர் யார்",
        "குடியரசுத் தலைவர் யார்",
        "தேசிய விலங்கு",
        "தேசிய கீதம்",
        "தேசிய பறவை",
        "தேசிய மலர்",
        "தமிழ்நாடு தலைநகர்",
        "இந்தியா தலைநகர்"
    ]
    
    words = text.split()
    
    # 1. Join adjacent words to check single-word target matching (e.g. "முதலமை செய்தேர்" -> "முதலமைச்சர்")
    i = 0
    while i < len(words) - 1:
        joined = words[i] + words[i+1]
        best_target = None
        best_score = 0.0
        # Check against single-word components of phrases/aliases
        targets = ["முதலமைச்சர்", "குடியரசுத்", "தலைவர்", "பிரதமர்"]
        for target in targets:
            score = levenshtein_similarity(joined, target)
            if score >= 0.65 and score > best_score:
                best_score = score
                best_target = target
        if best_target:
            phrase = f"{words[i]} {words[i+1]}"
            corrections.append((phrase, best_target, "Topic Keywords", best_score))
            words[i] = best_target
            # Remove the second word since it was merged
            words.pop(i+1)
            continue
        i += 1
        
    # 2. Token-aware multi-word pairing against ROLE_PHRASES
    i = 0
    while i < len(words) - 1:
        w1, w2 = words[i], words[i+1]
        best_target = None
        best_avg_score = 0.0
        
        for target in ROLE_PHRASES:
            t_words = target.split()
            if len(t_words) == 2:
                t1, t2 = t_words[0], t_words[1]
                score1 = levenshtein_similarity(w1, t1)
                score2 = levenshtein_similarity(w2, t2)
                
                if score1 >= 0.45 and score2 >= 0.45:
                    avg_score = (score1 + score2) / 2.0
                    if avg_score >= 0.5 and avg_score > best_avg_score:
                        best_avg_score = avg_score
                        best_target = target
                        
        if best_target:
            phrase = f"{w1} {w2}"
            if phrase != best_target:
                corrections.append((phrase, best_target, "Topic Keywords", best_avg_score))
                target_words = best_target.split()
                words[i] = target_words[0]
                words[i+1] = target_words[1]
            i += 2
            continue
        i += 1
    text = " ".join(words)
    
    # Priority 4: Single-word fuzzy matching
    words = text.split()
    entity_candidates = set(w for aliases in ENTITY_ALIASES.values() for a in aliases for w in a.split())
    topic_candidates = set(w for aliases in TOPIC_ALIASES.values() for a in aliases for w in a.split())
    
    for idx, w in enumerate(words):
        if w in GREETINGS_KEYWORDS:
            continue
            
        # Skip if already exactly matching an alias or canonical target
        if w in entity_candidates or w in topic_candidates:
            continue
            
        # Try entity match
        matched, score = fuzzy_match_word(w, entity_candidates, 0.7)
        if matched:
            corrections.append((w, matched, "Entity Keywords", score))
            words[idx] = matched
            continue
            
        # Try topic match
        matched, score = fuzzy_match_word(w, topic_candidates, 0.65)
        if matched:
            corrections.append((w, matched, "Topic Keywords", score))
            words[idx] = matched
            
    return " ".join(words), corrections


def classify_and_clean_query(normalized_text):
    """
    Orchestrates the Context-Aware routing flow.
    Performs intent detection and executes only the required cleaner route.
    """
    # 1. Exit detection (highest priority)
    exit_text, exit_corr = clean_exit(normalized_text)
    if is_exit_command(exit_text):
        return "Exit Command", exit_text, None, None, "Exit Handler", exit_corr

    # 2. Greeting detection (executed immediately after Exit detection, protected without fuzzy mapping)
    for kw in GREETINGS_KEYWORDS:
        if kw in normalized_text:
            return "Greeting", normalized_text, None, None, "Local Greeting Response", []

    # 3. Math check (fuzzy matches numbers ONLY after Math context is confirmed)
    has_math_context = False
    math_ops = ["கூட்டல்", "கழித்தல்", "பெருக்கல்", "வகுத்தல்", "பிளஸ்", "மைனஸ்"]
    math_nums = list(TAMIL_NUMBERS_MAP.keys())
    
    if any(c in normalized_text for c in "0123456789+-*/"):
        has_math_context = True
    else:
        words = normalized_text.split()
        for w in words:
            for num_word in math_nums:
                if w.startswith(num_word):
                    has_math_context = True
                    break
            if has_math_context:
                break
            if w in math_ops or w in math_nums:
                has_math_context = True
                break
            for math_w in math_ops + math_nums:
                if levenshtein_similarity(w, math_w) >= 0.7:
                    has_math_context = True
                    break
            if has_math_context:
                break

    if has_math_context:
        math_text, math_corr = clean_math(normalized_text)
        cleaned = math_text
        for word, digit in TAMIL_NUMBERS_MAP.items():
            cleaned = cleaned.replace(word, digit)
        cleaned = cleaned.replace("கூட்டல்", "+").replace("கழித்தல்", "-").replace("பெருக்கல்", "*").replace("வகுத்தல்", "/")
        cleaned = cleaned.replace("பிளஸ்", "+").replace("மைனஸ்", "-")
        cleaned = "".join(cleaned.split())
        if re.search(r"\d+\s*[\+\-\*/]\s*\d+", cleaned):
            return "Math Question", math_text, None, None, "Math Evaluator", math_corr

    # 4. Identity check
    has_identity_context = False
    identity_words = set(w for aliases in IDENTITY_ALIASES.values() for a in aliases for w in a.split())
    for w in normalized_text.split():
        if w in identity_words:
            has_identity_context = True
            break
        for id_w in identity_words:
            if levenshtein_similarity(w, id_w) >= 0.7:
                has_identity_context = True
                break
        if has_identity_context:
            break
            
    if has_identity_context:
        id_text, id_corr = clean_identity(normalized_text)
        for group_name, keywords in IDENTITY_ALIASES.items():
            if any(kw in id_text for kw in keywords):
                return "Identity Question", id_text, None, None, "Identity Handler", id_corr

    # 5. Fact check (Knowledge Guard)
    et_text, et_corr = clean_entity_topic(normalized_text)
    detected_entity = None
    for entity, aliases in ENTITY_ALIASES.items():
        if any(alias in et_text for alias in aliases):
            detected_entity = entity
            break

    detected_topic = None
    for topic, aliases in TOPIC_ALIASES.items():
        if any(alias in et_text for alias in aliases):
            detected_topic = topic
            break

    if detected_entity and detected_topic:
        ans = ENTITY_TOPIC_DB.get((detected_entity, detected_topic))
        if ans:
            return "Fact Question", et_text, detected_entity, detected_topic, "Knowledge Guard", et_corr
        else:
            return "Fact Question", et_text, detected_entity, detected_topic, "Ollama", et_corr

    # 6. Conversation or Unknown check -> No cleaner run, directly to Ollama
    chat_keywords = ["எப்படி", "நன்றி", "நன்றாக", "நலம்", "சுகம்", "நல்லது", "ஓகே", "சரி"]
    if any(kw in normalized_text for kw in chat_keywords):
        return "Conversation", normalized_text, None, None, "Ollama", []

    return "Unknown", normalized_text, None, None, "Ollama", []


def handle_greeting_query(cleaned_text):
    """
    Generate natural and conversational responses for greetings locally.
    """
    text = cleaned_text.lower()
    if "காலை வணக்கம்" in text or "காலைவணக்கம்" in text:
        return "காலை வணக்கம்! உங்கள் நாள் இனிதாக அமைய வாழ்த்துகள். என்ன உதவி வேண்டும்?"
    elif "மாலை வணக்கம்" in text or "மாலைவணக்கம்" in text:
        return "மாலை வணக்கம்! இன்று எப்படி உதவலாம்?"
    elif "வணக்கம்" in text:
        return "வணக்கம்! உங்களுக்கு என்ன உதவி வேண்டும்?"
    elif "ஹாய்" in text or "hi" in text:
        return "ஹாய்! என்ன உதவி வேண்டும்?"
    elif "ஹலோ" in text or "hello" in text:
        return "ஹலோ! உங்களுக்கு என்ன உதவி செய்யலாம்?"
    return "வணக்கம்! உங்களுக்கு என்ன உதவி வேண்டும்?"


def lookup_knowledge_guard(cleaned_text):
    """
    Detect entity and topic using centralized aliases databases and lookup answers.
    """
    detected_entity = None
    for entity, aliases in ENTITY_ALIASES.items():
        if any(alias in cleaned_text for alias in aliases):
            detected_entity = entity
            break

    detected_topic = None
    for topic, aliases in TOPIC_ALIASES.items():
        if any(alias in cleaned_text for alias in aliases):
            detected_topic = topic
            break

    if detected_entity and detected_topic:
        answer = ENTITY_TOPIC_DB.get((detected_entity, detected_topic))
        if answer:
            return answer, detected_entity, detected_topic
            
    # Resolve greeting query conversational fallbacks locally
    text_lower = cleaned_text.lower()
    for kw in GREETINGS_KEYWORDS:
        if kw in text_lower:
            return handle_greeting_query(cleaned_text), None, None

    return None, None, None


def evaluate_math_expression(cleaned_text):
    """
    Safely evaluate arithmetic expression (+ - * /) from Tamil input
    converting Tamil number words to digits first. No eval().
    """
    cleaned = cleaned_text
    for word, digit in TAMIL_NUMBERS_MAP.items():
        cleaned = cleaned.replace(word, digit)
        
    cleaned = cleaned.replace("கூட்டல்", "+").replace("கழித்தல்", "-").replace("பெருக்கல்", "*").replace("வகுத்தல்", "/")
    cleaned = cleaned.replace("பிளஸ்", "+").replace("மைனஸ்", "-")
    cleaned = "".join(cleaned.split())
    
    match = re.search(r"(\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(\d+(?:\.\d+)?)", cleaned)
    if match:
        num1_str, op, num2_str = match.groups()
        try:
            num1 = float(num1_str) if "." in num1_str else int(num1_str)
            num2 = float(num2_str) if "." in num2_str else int(num2_str)
            
            if op == "+":
                res = num1 + num2
            elif op == "-":
                res = num1 - num2
            elif op == "*":
                res = num1 * num2
            elif op == "/":
                if num2 == 0:
                    return "பூஜ்ஜியத்தால் வகுக்க முடியாது."
                res = num1 / num2
                if isinstance(res, float) and res.is_integer():
                    res = int(res)
                elif isinstance(res, float):
                    res = round(res, 4)
            return f"இதன் விடை {res} ஆகும்."
        except Exception:
            return None
    return None


def handle_identity_query(cleaned_text):
    """
    Match normalized text against Identity keyword groups.
    """
    for group_name, keywords in IDENTITY_ALIASES.items():
        if any(kw in cleaned_text for kw in keywords):
            if group_name == "உன் பெயர்":
                return "என் பெயர் தமிழ் குரல் AI உதவியாளர்."
            elif group_name == "நீ யார்":
                return "நான் உங்களுக்காக உருவாக்கப்பட்ட ஒரு தமிழ் குரல் உதவியாளர்."
            elif group_name == "நீ என்ன செய்ய முடியும்":
                return "நான் உங்கள் கேள்விகளுக்கு தமிழில் பதில் கூறவும், உரையாடவும் முடியும்."
            elif group_name == "உன்னை யார் உருவாக்கினார்":
                return "என்னை கூகுள் டீப்மைண்ட் குழுவினர் உருவாக்கியுள்ளனர்."
    return None


def classify_question(normalized_text):
    """
    Classify the query using normalized text.
    Returns: category, matched_entity, matched_topic
    """
    category, cleaned_text, entity, topic, decision, corrections = classify_and_clean_query(normalized_text)
    return category, entity, topic


def validate_ollama_response(text, category=None, user_text=None):
    """
    Validate response to reject empty text, punctuation, repeated symbols,
    consecutive repeating words, JSON fragments, or meaningless numbers.
    Also rejects obvious hallucinations, contradictions, and English fallback responses.
    """
    fallback = "மன்னிக்கவும். சரியான பதிலை உருவாக்க முடியவில்லை. தயவுசெய்து மீண்டும் கேளுங்கள்."
    if not text:
        return fallback
    cleaned = text.strip()
    if len(cleaned) == 0:
        return fallback

    # Punctuation only check
    punctuation_chars = set(string.punctuation + ".,!?;:\"'()[]{}<>*-+=_\\/|~`@#$%^&")
    if all(c in punctuation_chars or c.isspace() for c in cleaned):
        return fallback

    # Repeated symbols check (3 or more consecutive identical symbols)
    for i in range(len(cleaned) - 2):
        if cleaned[i] in punctuation_chars and cleaned[i] == cleaned[i+1] == cleaned[i+2]:
            return fallback

    # Repeated consecutive words check
    words = cleaned.split()
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            return fallback

    # JSON or markdown system check
    if "{" in cleaned and "}" in cleaned:
        if any(kw in cleaned for kw in ('"message"', '"content"', '"role"', '"response"')):
            return fallback
            
    if "```" in cleaned or "<system>" in cleaned:
        return fallback

    # Contradiction / Unrelated check if contexts are provided
    if category and user_text:
        user_lower = user_text.lower()
        cleaned_lower = cleaned.lower()
        
        # Unrelated entities/hallucinations check
        if any(kw in user_lower for kw in ["இந்தியா", "tamil", "தமிழ்"]):
            unrelated_entities = ["அலாஸ்கா", "alaska", "ஆப்பிரிக்கா", "africa", "அமெரிக்கா", "america"]
            if any(ent in cleaned_lower for ent in unrelated_entities):
                return fallback

        # Math question contradiction check
        if category == "Math Question":
            if not any(c.isdigit() for c in cleaned) and not any(w in cleaned for w in ["விடை", "பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது", "பத்து"]):
                return fallback

        # Fact question contradiction check (e.g. saying I don't know when a fact is expected)
        if category == "Fact Question":
            donotknow_phrases = ["தெரியாது", "தெரியவில்லை", "மன்னிக்கவும்", "பதில் இல்லை"]
            if any(phrase in cleaned for phrase in donotknow_phrases):
                return fallback

    # English-heavy response check (if more than 40% of words are English)
    english_word_pattern = re.compile(r"^[a-zA-Z]+$")
    english_count = sum(1 for w in words if english_word_pattern.match(w.strip(string.punctuation)))
    if len(words) > 0 and (english_count / len(words)) > 0.4:
        return fallback

    return cleaned


# ============================================================
# Ollama Response
# ============================================================


def ask_ollama(user_text):
    """
    Get a Tamil response from Ollama llama3.2 using local HTTP API.
    Maintains conversation history safely (only updates on success).
    """
    global conversation_history

    logger.info("Preparing request to Ollama...")

    # Construct temporary message list for payload to prevent early pollution
    messages_to_send = conversation_history + [{"role": "user", "content": user_text}]

    # Limit history context length (system prompt + last 10 turns = 20 messages)
    max_history_turns = 10
    if len(messages_to_send) > (max_history_turns * 2 + 1):
        messages_to_send = [messages_to_send[0]] + messages_to_send[-(max_history_turns * 2):]

    try:
        print("[SYSTEM] Thinking...")
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages_to_send,
            "stream": False
        }

        # Set reasonable timeouts using centralized config
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()

        response_data = response.json()
        
        # Check for invalid response format or missing fields
        if not isinstance(response_data, dict) or "message" not in response_data or "content" not in response_data["message"]:
            logger.error("Invalid response JSON format received from Ollama.")
            print("[ERROR] Invalid response JSON format from Ollama.")
            return "மன்னிக்கவும். தற்போது பதில் உருவாக்க முடியவில்லை."

        answer = response_data["message"]["content"].strip()

        if answer:
            # Update the global conversation history only after a successful response
            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": answer})
            
            # Maintain sliding window on global history
            if len(conversation_history) > (max_history_turns * 2 + 1):
                conversation_history = [conversation_history[0]] + conversation_history[-(max_history_turns * 2):]
                
            logger.info("Ollama response generated and conversation history updated.")
            return answer

        logger.warning("Ollama response content was empty.")
        return "மன்னிக்கவும். தற்போது பதில் உருவாக்க முடியவில்லை."

    except requests.RequestException as e:
        logger.error(f"Ollama HTTP request failed: {e}", exc_info=True)
        print(f"[ERROR] Ollama HTTP request failed: {e}")
        return "மன்னிக்கவும். தற்போது பதில் உருவாக்க முடியவில்லை."
    except Exception as e:
        logger.error("Unexpected error in ask_ollama", exc_info=True)
        print(f"[ERROR] Unexpected error in ask_ollama: {e}")
        return "மன்னிக்கவும். தற்போது பதில் உருவாக்க முடியவில்லை."


# ============================================================
# Cartesia Text-to-Speech
# ============================================================


def generate_speech(client, transcript):
    """Generate speech with Cartesia SDK v3.2.0 and save to reply.wav."""
    
    # Stop and unload any previous audio file to release the read stream lock
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        logger.info("Ensured previous audio playback is stopped and unloaded.")
    except Exception as e:
        logger.warning(f"Failed to stop/unload pygame music before generation: {e}", exc_info=True)

    # Wait 80 ms to allow Windows OS to release the file handle
    time.sleep(0.08)

    # Attempt to delete the existing output file before writing to it
    if os.path.exists(OUTPUT_FILE):
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                os.remove(OUTPUT_FILE)
                logger.info(f"Successfully deleted existing file {OUTPUT_FILE} (attempt {attempt}).")
                break
            except PermissionError as pe:
                logger.warning(
                    f"Permission denied deleting {OUTPUT_FILE} (attempt {attempt}/{max_retries}). Retrying in 100ms..."
                )
                if attempt == max_retries:
                    logger.error(f"Failed to delete {OUTPUT_FILE} after {max_retries} attempts.")
                    raise pe
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Unexpected error when removing file {OUTPUT_FILE}: {e}", exc_info=True)
                raise e

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


def play_wav(file_path):
    """Play a WAV file using pygame."""

    if not os.path.exists(file_path):
        logger.error(f"Audio file not found for playback: {file_path}")
        print(f"[ERROR] Audio not found: {file_path}")
        return

    try:
        logger.info("Loading and playing audio file...")
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        logger.info("Audio playback finished.")

    except Exception as e:
        logger.error("Pygame audio playback failed", exc_info=True)
        print("[ERROR] Unable to play audio")
        print(e)
    finally:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            logger.info("Pygame music stream stopped and unloaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to unload pygame mixer music: {e}", exc_info=True)


def speak(text):
    """
    Convert text to speech using Cartesia and play reply.wav.
    """

    if not text:
        return

    try:
        logger.info("Converting text to speech using Cartesia...")
        print(f"[AI] {text}")
        print("[SYSTEM] Generating speech with Cartesia...")

        generate_speech(cartesia_client, text)

        logger.info("TTS generation complete. Playing output audio...")
        print(f"[SYSTEM] Saved: {OUTPUT_FILE}")
        play_wav(OUTPUT_FILE)

    except Exception as e:
        logger.error("Unable to generate or play speech with Cartesia", exc_info=True)
        print("[ERROR] Unable to generate or play speech")
        print(e)


# ============================================================
# Main Program
# ============================================================


def is_exit_command(text):
    """
    Check if the user input matches exit commands or common mistakes,
    supporting specific polite forms, Levenshtein distance matching,
    and ignoring negated exit commands.
    """
    exit_candidates = []
    for aliases in EXIT_ALIASES.values():
        exit_candidates.extend(aliases)
        
    punctuation_chars = '.,!?;:"\'()[]{}<>*-+=_\\/|~`@#$%^&'
    clean_text = text.lower()
    for char in punctuation_chars:
        clean_text = clean_text.replace(char, ' ')
        
    words = clean_text.split()
    
    # Fuzzy match exit commands (restricted only to exit list candidates)
    for idx, w in enumerate(words):
        if w not in exit_candidates:
            matched_word, score = fuzzy_match_word(w, exit_candidates, 0.75)
            if matched_word:
                print_fuzzy_log("Exit Keywords", w, matched_word, score)
                words[idx] = matched_word
                
    # Map exit command aliases to canonical keys
    for idx, w in enumerate(words):
        for canonical, aliases in EXIT_ALIASES.items():
            if w in aliases:
                words[idx] = canonical
                break
                
    POLITE_FORMS = ("நிறுத்துங்கள்", "நிறுத்துங்க", "வெளியேறுங்கள்", "வெளியேறுங்க")
    NEGATION_SUFFIXES = ("ாதே", "ாதீர்கள்", "வேண்டாம்", "கூடாது")
    NEGATION_WORDS = ("வேண்டாம்", "கூடாது", "இல்லை")
    
    for i, word in enumerate(words):
        # 1. Negation detection
        is_neg = False
        for suffix in NEGATION_SUFFIXES:
            if word.endswith(suffix):
                is_neg = True
                break
        
        if not is_neg and i + 1 < len(words):
            if words[i+1] in NEGATION_WORDS:
                is_neg = True
                
        if is_neg:
            continue

        # 2. Check polite forms / canonical words
        if word in POLITE_FORMS or word in ("வெளியேறு", "நிறுத்து", "exit", "bye"):
            return True

    return False


def ensure_ollama_running():
    """
    Check if the Ollama server is running.
    If not, attempt to start it using subprocess.
    """
    try:
        # Check if server is already running
        requests.get(OLLAMA_BASE_URL, timeout=2)
        logger.info("Ollama server is active and online.")
        return True
    except requests.RequestException:
        logger.warning("Ollama server is offline. Attempting to start it via subprocess...")
        print("[SYSTEM] Ollama server is offline. Attempting to start it...")
        try:
            # Start ollama serve in the background (platform-independent flags)
            creationflags = 0
            if os.name == 'nt':
                import subprocess as sp
                creationflags = sp.CREATE_NO_WINDOW
                
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags
            )
            # Wait for it to become ready (up to 10 seconds)
            for _ in range(10):
                time.sleep(1)
                try:
                    requests.get(OLLAMA_BASE_URL, timeout=1)
                    logger.info("Ollama server started successfully via subprocess.")
                    print("[SYSTEM] Ollama server started successfully.")
                    return True
                except requests.RequestException:
                    continue
        except Exception as e:
            logger.error("Could not start Ollama server automatically", exc_info=True)
            print(f"[WARNING] Could not start Ollama server automatically: {e}")
            
    logger.error("Ollama server is offline and could not be started.")
    print("[ERROR] Ollama server is offline. Please make sure Ollama is running.")
    return False


def main():
    logger.info("Starting Voice AI main execution flow...")
    if not CARTESIA_API_KEY:
        logger.error("CARTESIA_API_KEY environment variable is missing.")
        print("[ERROR] Set the CARTESIA_API_KEY environment variable.")
        return

    # Check / start Ollama local server
    ensure_ollama_running()

    # Initialize pygame mixer only once at startup
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        logger.info("Pygame mixer initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize pygame mixer", exc_info=True)
        print(f"[ERROR] Failed to initialize pygame mixer: {e}")
        return

    try:
        speak("வணக்கம். நான் உங்கள் தமிழ் குரல் உதவியாளர்.")

        while True:
            audio_path = record_audio()
            if not audio_path:
                logger.warning("Microphone capture failed. Retrying in 2 seconds...")
                print("[SYSTEM] Recording failed, retrying in 2 seconds...")
                time.sleep(2)
                continue

            user_text = speech_to_text(audio_path)
            user_text = normalize_tamil_text(user_text)

            if not user_text:
                speak("மன்னிக்கவும். மீண்டும் சொல்லுங்கள்.")
                continue

            # Normalize query first
            normalized_user_text = normalize_query(user_text)

            # Apply Context-Aware Router
            category, cleaned_user_text, entity, topic, decision, corrections = classify_and_clean_query(normalized_user_text)

            # Commit fuzzy match logs only once for the selected route
            for original, matched, category_name, score in corrections:
                print_fuzzy_log(category_name, original, matched, score)

            if category == "Exit Command" or is_exit_command(cleaned_user_text):
                speak("நன்றி. மீண்டும் சந்திப்போம்.")
                logger.info("Exit command detected. Terminating conversation loop...")
                break

            response = None
            
            if category == "Greeting":
                response, ent, top = lookup_knowledge_guard(cleaned_user_text)
                if response is None:
                    response = "வணக்கம்! நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?"
                    
            elif category == "Identity Question":
                response = handle_identity_query(cleaned_user_text)
                
            elif category == "Math Question":
                response = evaluate_math_expression(cleaned_user_text)
                
            elif category == "Fact Question":
                response, ent, top = lookup_knowledge_guard(cleaned_user_text)

            # Print routing logs
            print("\n[ROUTER]")
            print(f"Original Query   : {user_text}")
            print(f"Normalized Query : {cleaned_user_text}")
            print(f"Category         : {category}")
            print(f"Matched Entity   : {entity if entity else 'None'}")
            print(f"Matched Topic    : {topic if topic else 'None'}")
            print(f"Decision         : {decision}")
            
            logger.info(
                f"[ROUTER] Original: {user_text} | Normalized: {cleaned_user_text} | Category: {category} | "
                f"Entity: {entity} | Topic: {topic} | Decision: {decision}"
            )

            # If no local response was generated, fallback to Ollama
            if response is None:
                response = ask_ollama(user_text)
                # Apply validation filters to prevent speaking garbage/hallucinations
                response = validate_ollama_response(response, category, user_text)

            speak(response)

    finally:
        # Proper cleanup of pygame mixer
        try:
            pygame.mixer.quit()
            logger.info("Pygame mixer closed successfully.")
        except Exception as e:
            logger.warning(f"Failed to close pygame mixer: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application closed by KeyboardInterrupt.")
        print("\nProgram Closed.")
    except Exception as e:
        logger.critical("Fatal unexpected error occurred in application", exc_info=True)
        print(f"\n[FATAL] Unexpected error: {e}")
        raise
