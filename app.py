import streamlit as st
import cv2
import speech_recognition as sr
from deep_translator import GoogleTranslator
from langdetect import detect
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering
)
import edge_tts
import asyncio
import tempfile
import time

# ---------------- STREAMLIT ----------------

st.set_page_config(page_title="AI Vision Assistant")

st.title("🤖 AI Vision Voice Assistant")

# ---------------- LANGUAGE ----------------

language_option = st.selectbox(
    "Choose Language",
    ["English", "Hindi"]
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-vqa-base"
    )

    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    )

    return processor, model

processor, model = load_model()

# ---------------- SPEAK FUNCTION ----------------

async def generate_voice(text, voice, filename):

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice
    )

    await communicate.save(filename)

def speak(text, lang="en"):

    voice_map = {
        "en": "en-US-AriaNeural",
        "hi": "hi-IN-SwaraNeural"
    }

    voice = voice_map.get(
        lang,
        "en-US-AriaNeural"
    )

    temp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    filename = temp_audio.name

    temp_audio.close()

    try:

        asyncio.run(
            generate_voice(
                text,
                voice,
                filename
            )
        )

    except RuntimeError:

        loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)

        loop.run_until_complete(
            generate_voice(
                text,
                voice,
                filename
            )
        )

    # Streamlit audio playback
    with open(filename, "rb") as audio_file:

        audio_bytes = audio_file.read()

        st.audio(
            audio_bytes,
            format="audio/mp3"
        )

# ---------------- LISTEN FUNCTION ----------------

def listen():

    recognizer = sr.Recognizer()

    try:

        with sr.Microphone() as source:

            st.write("🎤 Listening...")

            recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=5
            )

            if language_option == "Hindi":

                text = recognizer.recognize_google(
                    audio,
                    language="hi-IN"
                )

                lang = "hi"

            else:

                text = recognizer.recognize_google(
                    audio,
                    language="en-US"
                )

                lang = "en"

            return text, lang

    except:

        return "", "en"

# ---------------- CAMERA ----------------

camera = cv2.VideoCapture(0)

frame_window = st.image([])

if "last_caption" not in st.session_state:
    st.session_state.last_caption = ""

if "running" not in st.session_state:
    st.session_state.running = True

stop = st.button("Stop Assistant")

if stop:
    st.session_state.running = False

# ---------------- MAIN LOOP ----------------

while st.session_state.running:

    success, frame = camera.read()

    if not success:

        st.error("Camera not working")

        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame_window.image(rgb)

    img = Image.fromarray(rgb)

    # ---------------- SCENE UNDERSTANDING ----------------

    scene_question = "Describe this scene in detail"

    inputs = processor(
        img,
        scene_question,
        return_tensors="pt"
    )

    output = model.generate(**inputs)

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    # ---------------- SPEAK SCENE ----------------

    if caption != st.session_state.last_caption:

        st.subheader("📸 Detected Scene")

        st.write(caption)

        speak(caption)

        st.session_state.last_caption = caption

    # ---------------- USER VOICE ----------------

    user_voice, lang = listen()

    if user_voice != "":

        st.subheader("🗣 User Said")

        st.write(user_voice)

        # Detect language

        try:

            detected_lang = detect(user_voice)

        except:

            detected_lang = "en"

        # Translate to English

        english_question = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(user_voice)

        # ---------------- AI ANSWER ----------------

        inputs = processor(
            img,
            english_question,
            return_tensors="pt"
        )

        output = model.generate(**inputs)

        answer = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        # Translate back

        final_answer = GoogleTranslator(
            source="en",
            target=detected_lang
        ).translate(answer)

        st.subheader("🤖 Assistant")

        st.write(final_answer)

        speak(final_answer, detected_lang)

    time.sleep(3)

camera.release()

cv2.destroyAllWindows()
