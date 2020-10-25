import pandas as pd  # pandas 설치
import numpy as np
import matplotlib.pyplot as plt
import keras
import pickle
import re
import urllib.request
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Embedding, Dense, LSTM, Dropout
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request

app = Flask(__name__)

app.debug = True

with open('word_to_index.pickle', 'rb') as fr:
    dict_loaded = pickle.load(fr)

loaded_model = load_model('predict_test_model_10-06.h5')
print(dict_loaded)
max_len = 50
dict_loaded = Tokenizer(num_words=10000, oov_token="<OOV>")


# print(test_sequences)
# print(dict_loaded)

def predict_sentence(subject_sentence):
    cng_subject_sentence = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', subject_sentence)
    dict_loaded.fit_on_texts(cng_subject_sentence)  # 토큰화
    fword_to_index = dict_loaded.texts_to_sequences([cng_subject_sentence])
    pad_new = pad_sequences(fword_to_index, maxlen=max_len)
    score = float(loaded_model.predict(pad_new))
    if (score > 0.5):
        result = "정확도 : {:.5f}%".format(score)
        return result

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/send')
def send():
    return render_template('post.html')

@app.route('/post', methods=['POST'])
def post():
    value = request.form['send']
    result = predict_sentence(value)
    return result

if __name__ == '__main__':
    app.run()