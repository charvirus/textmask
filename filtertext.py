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
import pickle

# model = Word2Vec.load('ko.bin')
# wv = model.wv

# 엑셀 파일 경로
excel_data = pd.read_excel('runningfile.xlsx'
                           , sheet_name='Sheet1')

# 첫번째 행의 머리말을 따옴
# print(excel_data.columns)
# print(excel_data['내용'])
# 행의 개수 출력
# print(len(excel_data))
# excel['분류'].value_counts().plot(kind='bar')
# plt.show()
# print(excel.groupby('분류').size().reset_index(name = 'count'))


# 정규 표현식을 통한 한글 외 문자 제거
excel_data['내용'] = excel_data['내용'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

# print('댓글 최대 길이 : ',max(len(l) for l in excel_data['내용']))
# print('댓글 평균 길이 : ',sum(len(l) for l in excel_data['내용'])/len(excel_data['내용']))

# 문자 제거후 내용 출력
# for i in excel.index:
#     print(excel['내용'][i])

# 불용어
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다'
]

excel_data['분류'] = excel_data['분류'].replace(['Clean', 'Bad'], [0, 1])

max_words = 1000
maxlen = 50
X_data = excel_data['내용']
Y_data = excel_data['분류']

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_data)  # 토큰화
word_to_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(X_data)  # 단어를 숫자값, 인덱스로 변환하여 저장

print(word_to_index)  # 단어의 인덱스와 숫자 값을 보여줌
# print(X_data[:3])

flat_X_data = np.array(X_data).flatten().tolist()
# 여기 밑부터 막힘 학습을 시켜야하는데 어떻게 코딩을 해야할 지 모름 (10-06)

x = sequence.pad_sequences(sequences, maxlen)
# print(x.shape)
# print(x)

keras.backend.clear_session()
model = Sequential()
model.add(Embedding(max_words, 115))
model.add(LSTM(115))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# batch_size는 변수로 지정할것 예를들어 데이터가 1000개면 그것의 10%정도로 할 것
history = model.fit(x, Y_data, epochs=200, batch_size=50).history

test = ['진짜 쓰레기들', '미친 또라이 새끼들']

test_sequences = tokenizer.texts_to_sequences(test)

print(test_sequences)

# plt.plot(history['loss'], label='loss')
# plt.legend()
# plt.show()
# plt.plot(history['acc'], label='acc')
# plt.legend()
# plt.show()

# 현재 모델을 파일로 따로 저장함 , (적중률이 높은 모델이면 저장할 것)
model.save("predict_test_model_10-06.h5")

with open('word_to_index.pickle', 'wb') as fw:
    pickle.dump(word_to_index, fw)

# vocab_size = len(word_to_index) + 1
# n_of_train = int(len(sequences) * 0.8)
# n_of_test = int(len(sequences) - n_of_train)
# X_data = sequences
# model = Sequential()
# model.add(Embedding(vocab_size, 32)) # 임베딩 벡터의 차원은 32
# model.add(SimpleRNN(32)) # RNN 셀의 hidden_size는 32
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(X_train, y_train, epochs=5, batch_size=1024, validation_split=0.2).history

# print('\nko.bin에 있는 내역이 엑셀 데이터에 있는지 찾아줌\n')
# okt = Okt()
# tokenized_data = []
# for sentence in excel_data['내용']:
#     temp_x = okt.morphs(sentence, stem=False)  # 토큰화
#     temp_x = [word for word in temp_x if not word in stopwords]
#     tokenized_data.append(temp_x)

# tokenized_data = []
# for sentence in excel['내용']:
#     temp_x = okt.morphs(sentence, stem=False)  # 토큰화
#     tokenized_data.append(temp_x)
#     for w in temp_x:
#         print(w, end=" ")
#     print()
# plt.hist([len(s) for s in tokenized_data], bins= 50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()
# model = Word2Vec(sentences=tokenized_data, size=100, window=5, min_count=5, workers=4, sg=0)
# for k in model.wv.vectors:
#     print(k)
# 완성된 임베딩 매트릭스의 크기 확인
# print(model.wv.vectors.shape)
# print(model.wv.most_similar("대가리"))
