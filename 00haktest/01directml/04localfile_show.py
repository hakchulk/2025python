import os
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_unzipped(path, kind='train'):
    """압축 해제된 MNIST 바이너리 파일에서 데이터를 로드합니다."""
    # 파일명 설정 (압축이 풀린 경우 보통 .gz가 없음)
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    # 레이블 읽기
    with open(labels_path, 'rb') as lbpath:
        # 헤더: 8바이트 (Magic Number 4바이트 + 데이터 개수 4바이트)
        labels = np.fromfile(lbpath, dtype=np.uint8, offset=8)

    # 이미지 읽기
    with open(images_path, 'rb') as imgpath:
        # 헤더: 16바이트 (Magic Number, 데이터 개수, 행, 열 각 4바이트씩)
        images = np.fromfile(imgpath, dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)

    return images, labels

# 1. 파일 경로 설정 (r을 붙여 역슬래시 인식 문제 방지)
base_path = r'c:\HAK\kd\2025python\00ai\data\MNIST\raw'

try:
    # 2. 데이터 불러오기
    train_images, train_labels = load_mnist_unzipped(base_path, kind='train')

    # 3. 시각화 (10개 출력)
    plt.figure(figsize=(12, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(train_images[i], cmap='gray')
        plt.title(f"Label: {train_labels[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    print(f"이미지 데이터 로드 완료! 크기: {train_images.shape}")

except FileNotFoundError as e:
    print(f"에러: 해당 경로에서 파일을 찾을 수 없습니다. 파일명을 다시 확인해주세요.\n{e}")