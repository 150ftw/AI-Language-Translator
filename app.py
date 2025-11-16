import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# --------------------------------------------
# PRELOAD MODEL AT STARTUP (Before UI renders)
# --------------------------------------------
@st.cache_resource
def preload_model():
    lang_map = {
        "English → Hindi": ("en", "hi"),
        "Hindi → English": ("hi", "en"),
        "English → Spanish": ("en", "es"),
        "Spanish → English": ("es", "en"),
        "English → French": ("en", "fr"),
        "French → English": ("fr", "en"),
    }

    preloaded = {}

    for label, (src, tgt) in lang_map.items():
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        preloaded[label] = (tokenizer, model)

    return preloaded, lang_map


# Load all models ONCE before user interacts
PRELOADED_MODELS, LANG_MAP = preload_model()


# --------------------------------------------
# Translation Function
# --------------------------------------------
def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs, max_length=200)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


# --------------------------------------------
# UI
# --------------------------------------------
st.set_page_config(page_title="Offline Language Translator", layout="centered")
st.title("🌍 Language Translator")

st.markdown("### Choose Language Pair")
choice = st.selectbox("Translation Type", list(LANG_MAP.keys()))

text = st.text_area("Enter text to translate")

tokenizer, model = PRELOADED_MODELS[choice]

if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter text.")
    else:
        output = translate_text(text, tokenizer, model)
        st.success("Translation:")
        st.write(output)