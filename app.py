import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model('model.h5', compile=False)

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_next_word = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_next_word:
            return word
    return "[Unknown Word]"

# Streamlit app
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="📝",
    layout="centered",
)

# App header
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Next Word Prediction</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Type a sentence below and get the next predicted word!</p>",
    unsafe_allow_html=True,
)

# User input
text = st.text_input("Enter your text:", "I am")

# Prediction button
if st.button("Predict Next Word"):
    max_seq_len = model.input_shape[1] + 1
    next_word = predict(model, tokenizer, text, max_seq_len)
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <h3 style="color: #4CAF50; display: inline;">Predicted Word: </h3>
            <h3 style="color: #FF5722; display: inline;"><b>{}</b></h3>
        </div>
        """.format(next_word),
        unsafe_allow_html=True,
    )
