import pandas as pd  # pandas 설치
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import urllib.request
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Embedding, Dense, LSTM, Dropout
from keras.preprocessing.sequence import pad_sequences

loaded_model = load_model('predict_test_model_10-06.h5')
max_len = 20


def predict_sentence(subject_sentence):
    ftoken = Tokenizer()
    ftoken.fit_on_texts(subject_sentence)  # 토큰화
    fword_to_index = ftoken.texts_to_sequences([subject_sentence])
    pad_new = pad_sequences(fword_to_index, maxlen=max_len)
    score = float(loaded_model.predict(pad_new))
    if (score > 0.5):
        print("정확도 : {:.2f}%".format(score * 100))


predict_sentence('미친 쓰레기들')
predict_sentence('가해자 인권 중심 집단답네')

predict_sentence('희망')
