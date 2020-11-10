import pandas as pd  # pandas 설치
import numpy as np
import keras
import re
import urllib.request
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Embedding, Dense, LSTM, Dropout
from keras.preprocessing.sequence import pad_sequences
import pickle
from matplotlib import pyplot as plt

# 엑셀 파일 경로
excel_data = pd.read_excel('runningfile.xlsx', sheet_name='Sheet1')

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

# 문자 제거후 내용 출력
# for i in excel.index:
#     print(excel['내용'][i])

excel_data['분류'] = excel_data['분류'].replace(['Clean', 'Bad'], [0, 1])

max_words = 10000
maxlen = 50
X_data = excel_data['내용']
print(X_data)
Y_data = excel_data['분류']

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_data)  # 토큰화
word_to_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(X_data)  # 단어를 숫자값, 인덱스로 변환하여 저장

print(word_to_index)  # 단어의 인덱스와 숫자 값을 보여줌
print(sequences)
flat_X_data = np.array(X_data).flatten().tolist()

x = sequence.pad_sequences(sequences, maxlen)

keras.backend.clear_session()
model = Sequential()
model.add(Embedding(max_words, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# batch_size는 변수로 지정할것 예를들어 데이터가 1000개면 그것의 10%정도로 할 것
history = model.fit(x, Y_data, epochs=200, batch_size=1000).history


# 현재 모델을 파일로 따로 저장함 , (적중률이 높은 모델이면 저장할 것)
model.save("predict_test_model_10-06.h5")

with open('tokenizer.pickle', 'wb') as fw:
    # pickle.dump(word_to_index, fw)
    pickle.dump(tokenizer, fw, protocol=pickle.HIGHEST_PROTOCOL)

plt.plot(history['acc'])
plt.plot(history['loss'])
plt.show()

