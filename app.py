import streamlit as st
import pygame
import cv2
import speech_recognition as sr
from deep_translator import GoogleTranslator
from langdetect import detect
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)
import edge_tts
import asyncio
import time
pygame.mixer.init()
st.set_page_config(page_title="AI Vision Assistant")

st.title(" AI Vision Voice Assistant")

# ---------------- LANGUAGE OPTION ----------------

language_option = st.selectbox(

    "Choose Language",

    [
        "English",
        "Hindi"
    ]
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model


processor, model = load_model()

import tempfile
import os

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

    # Create temporary mp3 file
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

    # Stop previous audio
    pygame.mixer.music.stop()

    # Load new audio
    pygame.mixer.music.load(filename)

    # Play automatically
    pygame.mixer.music.play()
# ---------------- LISTEN FUNCTION ----------------

def listen():

    recognizer = sr.Recognizer()

    try:

        with sr.Microphone() as source:

            st.write(" Listening...")

            recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=5
            )

            # Language based recognition
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

    # Generate caption
    inputs = processor(
        img,
        return_tensors="pt"
    )

    output = model.generate(**inputs)

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    # Speak only when changed
    if caption != st.session_state.last_caption:

        st.subheader(" Detected Scene")

        st.write(caption)

        speak(caption)

        st.session_state.last_caption = caption

    # Listen user
    user_voice, lang = listen()

    if user_voice != "":

        st.subheader(" User Said")

        st.write(user_voice)

        # Safe language detection
        try:

            detected_lang = detect(user_voice)

            supported_languages = [
                "en",
                "hi",
                "fr",
                "es",
                "de"
            ]

            if detected_lang not in supported_languages:
                detected_lang = "en"

        except:

            detected_lang = "en"

        # Translate to English
        english_question = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(user_voice)

        # Smart assistant response
        if "what did you say" in english_question.lower():

            answer = f"I said: {caption}"

        elif "what do you see" in english_question.lower():

            answer = f"I can see: {caption}"

        elif "hello" in english_question.lower():

            answer = "Hello, how can I help you?"

        else:

            answer = (
                f"You asked: {english_question}. "
                f"I can see {caption}"
            )

        # Translate back to user language
        final_answer = GoogleTranslator(
            source="en",
            target=detected_lang
        ).translate(answer)

        st.subheader(" Assistant")

        st.write(final_answer)

        # Speak answer
        speak(final_answer, detected_lang)

    time.sleep(3)

camera.release()
cv2.destroyAllWindows()