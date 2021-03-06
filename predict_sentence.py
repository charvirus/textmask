import keras
import pickle
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Embedding, Dense, LSTM, Dropout
from keras.preprocessing.sequence import pad_sequences
from flask import Flask

app = Flask(__name__)

app.debug = True

with open('word_to_index.pickle', 'rb') as fr:
    dict_loaded = pickle.load(fr)

loaded_model = load_model('predict_test_model_10-06.h5')
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

