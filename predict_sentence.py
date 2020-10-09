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

with open('word_to_index.pickle', 'rb') as fr:
    dict_loaded = pickle.load(fr)


loaded_model = load_model('predict_test_model_10-06.h5')
max_len = 30
print(dict_loaded)
dict_loaded = Tokenizer(num_words=1000, oov_token="<OOV>")
# print(dict_loaded)
test = ['진짜 쓰레기들', '미친 또라이 새끼들']
test_sequences = dict_loaded.texts_to_sequences(test)
# print(test_sequences)
# print(dict_loaded)


def predict_sentence(subject_sentence):
    dict_loaded.fit_on_texts(subject_sentence)  # 토큰화
    fword_to_index = dict_loaded.texts_to_sequences([subject_sentence])
    pad_new = pad_sequences(fword_to_index, maxlen=max_len)
    score = float(loaded_model.predict(pad_new))
    if (score > 0.5):
        print("정확도 : {:.2f}%".format(score * 100))


predict_sentence('멍청한 것들')
predict_sentence('또라이 중심 집단답네')
predict_sentence('민주당은 무슨 공산당이네 ㅋㅋㅋㅋ')