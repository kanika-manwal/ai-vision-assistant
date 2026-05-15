import streamlit as st
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
import tempfile

st.set_page_config(page_title="AI Vision Assistant")

st.title("🤖 AI Vision Voice Assistant")

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

    audio_file = open(filename, "rb")

    audio_bytes = audio_file.read()

    st.audio(
        audio_bytes,
        format="audio/mp3",
        autoplay=True
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

            # Language selection
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

# ---------------- IMAGE INPUT ----------------

image = st.camera_input("Take a picture")

if image is not None:

    img = Image.open(image).convert("RGB")

    st.image(img)

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

    st.subheader("📝 Detected Scene")

    st.write(caption)

    # Speak caption
    speak(caption)

    # Listen user
    user_voice, lang = listen()

    if user_voice != "":

        st.subheader("🗣 User Said")

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

        # Translate question
        english_question = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(user_voice)

        # Smart responses
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

        # Translate answer
        final_answer = GoogleTranslator(
            source="en",
            target=detected_lang
        ).translate(answer)

        st.subheader("🤖 Assistant")

        st.write(final_answer)

        # Speak automatically
        speak(final_answer, detected_lang)
