import re
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from flask import request
from flask import make_response
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Embedding, Dense, LSTM, Dropout
from keras.preprocessing.sequence import pad_sequences

import pickle

with open('word_to_index.pickle', 'rb') as fr:
    dict_loaded = pickle.load(fr)
g = tf.Graph()
session = tf.compat.v1.Session()
loaded_model = load_model('predict_test_model_10-06.h5')
max_len = 50
dict_loaded = Tokenizer(num_words=10000, oov_token="<OOV>")

def predict_sentence(subject_sentence):

    cng_subject_sentence = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', subject_sentence)
    dict_loaded.fit_on_texts(cng_subject_sentence)  # 토큰화
    fword_to_index = dict_loaded.texts_to_sequences([cng_subject_sentence])
    pad_new = pad_sequences(fword_to_index, maxlen=max_len)
    score = float(loaded_model.predict(pad_new))
    if score > 0.5:
        result = "정확도 : {:.5f}%".format(score)
        return result


app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

@app.route('/')
def index():
    f = open('chat.html')
    s = f.read()
    return s
    #return render_template('chat.html')

@app.route('/images/<path:path>')
def send_images(path):
    return send_from_directory('images', path)

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):

    msg = json["message"]

    print('received client id : ' + request.sid)
    socketio.emit('my response', json)

if __name__ == '__main__':

    samples = [
    "개새끼"
    ,"시발"
    ,"이게 말이 되냐?"
    ,"개소리하네"
    ,"미친놈들 천국이야"
    ,"시발 ㅋㅋㅋㅋㅋㅋㅋㅋ"
    ,"지랄하네"
    ,"멍청한 것들"
    ,"안녕하세요 혹시 거래 되시나요?"
    ,"네 가능합니다"]

    for s in samples :
        print(s,":", predict_sentence(s))


    socketio.run(app, debug=True)
