# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# #prediction function
# def predict_sentiment(review):
#     preprocessed_input=preprocess_text(review)

#     prediction=model.predict(preprocessed_input)

#     sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
#     return sentiment, prediction[0][0]


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# import streamlit as st
# ## streamlit app
# # Streamlit app
# st.title('IMDB Movie Review Sentiment Analysis')
# st.write('Enter a movie review to classify it as positive or negative.')

# # User input
# user_input = st.text_area('Movie Review')

# if st.button('Classify'):

#     preprocessed_input=preprocess_text(user_input)

#     ## MAke prediction
#     prediction=model.predict(preprocessed_input)
#     sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

#     # Display the result
#     st.write(f'Sentiment: {sentiment}')
#     st.write(f'Prediction Score: {prediction[0][0]}')
# else:
#     st.write('Please enter a movie review.')

import streamlit as st
from PIL import Image

# Streamlit app title and description
st.markdown(
    """
    <div style="background-color: black; padding: 10px; border-radius: 10px;">
        <h1 style="color: white; text-align: center;">IMDB Movie Review Sentiment Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; margin-top: 10px;">
        <p style="font-size: 18px; color: #333; text-align: center;">
            Enter a movie review to classify it as positive or negative.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# User input with a styled text area
user_input = st.text_area(
    'Movie Review', 
    height=150, 
    placeholder='Type your movie review here...',
    help="Enter the text of the movie review you want to analyze."
)

# Button to trigger classification
if st.button('Classify'):
    if user_input:
        # Preprocess input (you would need to define preprocess_text function)
        preprocessed_input = preprocess_text(user_input)

        # Make prediction (assuming you have a pre-trained model loaded as `model`)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the result with colors and boxes
        result_color = "black" if sentiment == 'Positive' else "#F44336"
        st.markdown(
            f"""
            <div style="background-color: {result_color}; padding: 10px; border-radius: 10px; margin-top: 10px;">
                <h2 style="color: white; text-align: center;">Sentiment: {sentiment}</h2>
                <p style="color: white; text-align: center;">Prediction Score: {prediction[0][0]:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color: #FFEB3B; padding: 10px; border-radius: 10px; margin-top: 10px;">
                <p style="font-size: 18px; color: #333; text-align: center;">
                    Please enter a movie review before classifying.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; margin-top: 10px;">
            <p style="font-size: 18px; color: #333; text-align: center;">
                Please enter a movie review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


