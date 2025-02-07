import streamlit as st
import google.generativeai as genai
import requests
import os
from PIL import Image
import io

# API configuration
genai.configure(api_key='AIzaSyAmOR3pcla6c61VwOf604M0dI4eGch8C7Q')
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
# hf_key = os.environ["HF_API_KEY"]
headers = {"Authorization": f"Bearer {'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'}"}

# Model Creation
translation_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="Translate from Tamil to English with the same number of words.",
)

content_generation_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="Generate a story based on the given text without missing the given context.",
)


summarizer = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="Give the key points of the given text in a single line without avoiding the context.",
)


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


def translate(tamil_text: str):
    response = translation_model.generate_content(tamil_text)
    return response.text


def generate_image(prompt: str):
    if len(prompt) >= 50:
        response = summarizer.generate_content(prompt)
        prompt = response.text
    image_bytes = query({"inputs": prompt})
    return Image.open(io.BytesIO(image_bytes))


def content_generation(translated_text):
    content = content_generation_model.generate_content(translated_text)
    return content.text


# Streamlit app
st.title("AI-Powered Translation and Content Generation")

tamil_text = st.text_input("Tamil text:")

if tamil_text:
    with st.spinner("Translating..."):
        translated_text = translate(tamil_text)
    st.markdown("**Translated:**")
    st.write(translated_text)

    with st.spinner("Generating image..."):
        img = generate_image(translated_text)

    st.markdown("**Image:**")
    st.image(img, width=400)

    with st.spinner("Writing story..."):
        content = content_generation(translated_text)
    st.markdown("**Explanation:**")
    st.write(content)
