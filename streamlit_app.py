# load dependencies
import re
import nltk
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import nltk
nltk.download(‘stopwords’)

from textblob import Word
from keras import backend as K
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# path of the model
MODEL_PATH = r"model.h5"
# maximum number of the allowed word in an input 
max_words = 500
# shape of input data passed for prediction
max_len = 20
# path of tokenizer file
tokenizer_file = 'tokenizer.pkl'

# load tokenizer
with open(tokenizer_file,'rb') as handle:
    tokenizer = pickle.load(handle)

# apply text cleaning to input data
def text_cleaning(line_from_column):
    text = line_from_column.lower()
    # Replacing the digits/numbers
    text = text.replace('d', '')
    # remove stopwords
    words = [w for w in text if w not in stopwords.words("english")]
    # apply stemming
    words = [Word(w).lemmatize() for w in words]
    # merge words 
    words = ' '.join(words)
    return text

# load the model
#@st.cache_data(allow_output_mutation=True)
@st.cache_resource
def Load_model():
    model = load_model("model.h5")
    model.summary() # included making it visible when the model is reloaded
    session = K.get_session()
    return model, session

#final execution flow
if __name__ == '__main__':
    st.title('Sarcasm Detection for News Headlines app')
    st.write('A simple app to detect sarcasm')
    st.subheader('Input the News Headline below')
    sentence = st.text_area('Enter your thread here',height=200)
    predict_btt = st.button('predict')
    model, session = Load_model()
    if predict_btt:
        clean_text = []
        K.set_session(session)
        i = text_cleaning(sentence)
        clean_text.append(i)
        sequences = tokenizer.texts_to_sequences(clean_text)
        data = pad_sequences(sequences, maxlen =  max_len)
        # st.info(data)
        prediction = model.predict(data)
        
        st.header('Prediction using word2vec model')
        if prediction[0] >= 0.90:
          st.warning('Sarcasm')
        else:
          st.success('Not sarcasm')

        # prediction = model.predict(data)
        # prediction_prob_negative = prediction[0][0]
        # prediction_prob_positive = prediction[0][1]
        # #prediction_prob_positive= prediction[0][2]
        # prediction_class = prediction.argmax(axis=-1)[0]
        # print(prediction.argmax())
        # st.header('Prediction using GRU model')
        # if prediction_class == 0:
        #   st.warning('Thread has negative sentiment')
        # #if prediction_class == 1:
        # #  st.success('Thread has neutral sentiment')
        # if prediction_class==1:
        #   st.success('Thread has positive sentiment')