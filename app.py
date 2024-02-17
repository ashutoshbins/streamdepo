import streamlit as st
from transformers import pipeline
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import textwrap
import spacy
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
model_path = os.path.join(os.path.dirname(__file__), "en_core_web_sm")
# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Load the tokenizer and label encoder for sentiment analysis
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)


nlp = spacy.load(model_path)
# Load the machine learning model using Keras
model = load_model('my_model.h5')

# Function to extract text from a PDF URL
def extract_text_from_pdf_url(url):
    response = requests.get(url)
    pdf_file = BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Replace non-breaking space with regular space
    text = text.replace('\xa0', ' ')

    return text

# Function to split text into lines with line breaks after every n words
def split_text_with_line_break(text, words_per_line=10):
    words = text.split()
    lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return "\n".join(lines)

# Define preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(tokens)

# Streamlit UI
def main():
    st.title('PDF Analyzer')
    pdf_url = st.text_input('Paste PDF URL:')
    if st.button('Analyze'):
        if not pdf_url.startswith('/supremecourt'):
            st.error("Invalid URL. The URL should start with '/supremecourt'.")
        else:
            full_url = 'https://main.sci.gov.in' + pdf_url
            text = extract_text_from_pdf_url(full_url)
            lower_text = text.lower()
            user_input = preprocess_text(lower_text)
            first_50_words = lower_text.split()[:50]

            if 'civil' in first_50_words and 'jurisdiction' in first_50_words:
                priority_message = 'Priority: Low since civil case'
            elif 'criminal' in first_50_words and 'jurisdiction' in first_50_words:
                priority_message = 'Priority: High since criminal case'
            else:
                priority_message = 'Priority: Not specified'

            user_input_sequence = tokenizer.texts_to_sequences([user_input])
            user_input_padded = pad_sequences(user_input_sequence, maxlen=100, padding='post')

            prediction = model.predict(user_input_padded)
            threshold = 0.5
            binary_prediction = (prediction > threshold).astype(int)[0, 0]
            inverted_prediction = label_encoder.inverse_transform([binary_prediction])[0]
            prediction_result = f'The prediction is: ({inverted_prediction})'

            chunks = textwrap.wrap(text, width=1024)
            summary = []
            for chunk in chunks:
                if len(chunk.split()) > 50:
                    summary.append(summarizer(chunk, max_length=50, min_length=25, do_sample=False)[0]['summary_text'])
                else:
                    summary.append(chunk)

            summarized_content = []
            for sentence in summary:
                lines = split_text_with_line_break(sentence)
                summarized_content.extend(lines.split('\n'))
            summarized_content.insert(0, priority_message)

            st.subheader('Analysis Results')
            st.write(f'PDF URL: {pdf_url}')
            st.write(prediction_result)
            for line in summarized_content:
                st.write(line)

if __name__ == '__main__':
    main()
