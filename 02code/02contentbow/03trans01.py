from sentence_transformers import SentenceTransformer, util

# all-MiniLM-L6-v2 : 문장 임베딩(Sentence Embedding) 분야에서 가장 대중적으로 사용되는 가성비 끝판왕 모델
# 'snunlp/KR-SBERT-V40K-klueNLI-augSTS' 한국어 성능이 검증된 모델로 변경 (SBERT-KR 등)
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

sentences = [
    "부산 여행 가고 싶다",
    "부산 바다 보고 싶다",
    "주식 시장이 어렵다"
]

embeddings = model.encode(sentences) # 문장 임베딩 벡터 생성
similarity = util.cos_sim(embeddings[0], embeddings[1]) # 코사인 유사도 계산
print(f"Similarity  {similarity}")

similarity = util.cos_sim(embeddings[0], embeddings[2]) # 코사인 유사도 계산
print(f"Similarity  {similarity}")
