
import pandas as pd # pandas 설치
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt



# 엑셀 파일 경로
excel = pd.read_excel('D:/Brandon/IT융합/3학년/2학기/인공지능/프로젝트/엑셀/인공지능 데이터수집 통합본(2).xlsx'
                      ,sheet_name='Sheet1')
print(excel.columns) # 첫번째 행의 머리말을 따옴
print(excel['내용'])
# for i in excel.index:
#     print(excel['내용'][i])

print(len(excel)) # 행의 개수 출력
# 정규 표현식을 통한 한글 외 문자 제거
excel['내용'] = excel['내용'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# 문자 제거후 내용 출력
for i in excel.index:
    print(excel['내용'][i])

# 불용어
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다'
             ,'들과','들이']

okt = Okt()
tokenized_data = []
for sentence in excel['내용']:
    temp_x = okt.morphs(sentence,stem=True) # 토큰화
    temp_x = [word for word in temp_x if not word in stopwords] # 불용어 제거
    tokenized_data.append(temp_x)

# 댓글 길이 분포 확인
print('댓글 최대 길이 : ',max(len(l) for l in tokenized_data))
print('댓글 평균 길이 : ', sum(map(len,tokenized_data))/len(tokenized_data))

# plt.hist([len(s) for s in tokenized_data], bins= 50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()

model = Word2Vec(sentences=tokenized_data, size=100,window=5,min_count=5,workers=4,sg=0)
# 완성된 임베딩 매트릭스의 크기 확인
# print(model.wv.vectors.shape)

print(model.wv.most_similar("새끼"))