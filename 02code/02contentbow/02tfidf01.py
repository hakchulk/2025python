# TF-IDF(Term Frequency-Inverse Document Frequency)는 텍스트 마이닝에서 가장 기본적이면서도 강력한 단어 가중치 산출 방식
# TF-IDF의 핵심 원리인 "흔한 단어는 가볍게, 희귀한 단어는 무겁게

# 텍스트 데이터를 머신러닝 모델이 이해할 수 있도록 수치형 벡터로 변환해주는 도구
# 각 단어가 특정 문서 내에서 얼마나 중요한지를 나타내는 TF-IDF 값을 계산
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns # 아름답고 복잡한 통계 그래픽 데이터 시각화 라이브러리
import matplotlib.pyplot as plt
import pandas as pd

import platform

# 운영체제별 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic') # 윈도우 (맑은 고딕)
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic') # 맥 (애플고딕)

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

documents = [
    "부산 여행 바다 맛집",
    "부산 해변 바다 산책",
    "서울 맛집 데이트",
    "제주도 여행 자연"
]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)
# print(vectorizer.vocabulary_)
# print(vectorizer.get_feature_names_out()) # 단어 사전 출력
# print(tfidf.toarray()) # 단어 사전과 각 문서의 TF-IDF 벡터 출력
# print(tfidf.toarray()[0])
# print(tfidf.toarray()[1])
# print(tfidf.toarray()[2])
# print(tfidf.toarray()[3])

# df_tfidf =  pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out(), index=[f'문서{i}' for i in range(len(documents))])
df_tfidf =  pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out(), index=[f'{i}.{documents[i]}' for i in range(len(documents))])
plt.figure(figsize=(10, 6))
sns.heatmap(df_tfidf
    , annot=True # 셀 안에 값 표시
    , cmap='YlGnBu' # 색상 맵
    # ,cmap='Reds' # 색상 맵
    )
plt.title('TF-IDF Heatmap')
plt.xlabel('단어')
plt.ylabel('문서')
plt.show()