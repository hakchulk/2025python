import matplotlib.pyplot as plt
import tensorflow as tf
# 1. MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist

# 1. MNIST 데이터셋 로드 (학습용, 테스트용)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2. 시각화를 위한 설정
plt.figure(figsize=(12, 5))

# 3. 10개의 이미지를 반복문을 통해 출력
for i in range(10):
    plt.subplot(2, 5, i + 1) # 2행 5열 구조
    plt.imshow(train_images[i], cmap='gray') # 그레이스케일로 이미지 출력
    plt.title(f"Label: {train_labels[i]}")   # 정답(Label) 표시
    plt.axis('off') # 축 숨기기

plt.tight_layout()
plt.show()

# 데이터 정보 출력
print(f"학습 이미지 모양: {train_images.shape}")
print(f"학습 레이블 모양: {train_labels.shape}")