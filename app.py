# app.py
import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import re
import string
import nltk
from nltk.corpus import stopwords

# --- Preprocessing Setup ---

# Download stopwords (if not already done)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# URL pattern
url_pattern = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)

# Punctuation pattern
punct_pattern = re.compile('[%s]' % re.escape(string.punctuation))

# Chat words mapping
chat_words = {
    "u": "you", "r": "are", "ur": "your", "btw": "by the way", "idk": "i do not know",
    "lol": "laughing out loud", "thx": "thanks", "pls": "please", "plz": "please",
    "omg": "oh my god", "gonna": "going to", "wanna": "want to", "im": "i am",
    "dont": "do not", "cant": "cannot", "doesnt": "does not", "isnt": "is not",
    "wasnt": "was not", "shouldnt": "should not", "couldnt": "could not",
    "wouldnt": "would not", "ive": "i have", "id": "i would", "didnt": "did not",
    "hru": "how are you", "brb": "be right back", "ttyl": "talk to you later",
    "gr8": "great", "b4": "before", "imo": "in my opinion", "fyi": "for your information",
    "smh": "shaking my head", "lmk": "let me know", "np": "no problem",
    "ty": "thank you", "yw": "you are welcome"
}

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    
    # lowercase
    text = text.lower()
    
    # remove html
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # remove urls
    text = url_pattern.sub('', text).strip()
    
    # remove punctuation
    text = punct_pattern.sub('', text)
    
    # expand chatwords
    words = text.split()
    words = [chat_words.get(w, w) for w in words]
    
    # remove stopwords
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words)

# --- Load Model and Tokenizer ---
model = tf.keras.models.load_model("sentiment_lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

# --- Streamlit App ---
st.title("Movie Review Sentiment Analysis")
review_input = st.text_area("Enter your review:")

if st.button("Predict"):
    if review_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        # preprocess before tokenizing
        cleaned_review = preprocess_text(review_input)
        seq = tokenizer.texts_to_sequences([cleaned_review])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        prob = model.predict(padded)[0][0]
        label = "Positive" if prob > 0.5 else "Negative"
        st.write(f"Predicted Sentiment: **{label}**")
        st.write(f"Confidence: {prob:.2f}")
