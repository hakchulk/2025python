import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()

fruits = np.load('data/fruits_300.npy') # 3d
fruits_2d = fruits.reshape(-1, 100 * 100) # 2d

# print(fruits_2d.shape)

km = KMeans(n_clusters=3, random_state=42) # 3개의 그룹으로 자동 분류
km.fit(fruits_2d)
print(km.labels_)
print(np.unique(km.labels_, return_counts=True))

# 각 클러스터의 중심(평균 이미지) 시각화
# fig, axs = plt.subplots(1, 3, figsize=(10, 3))
# for i in range(3):
#     # 10,000차원을 다시 100x100 이미지로 복원
#     center_img = km.cluster_centers_[i].reshape(100, 100)    
#     axs[i].imshow(center_img, cmap='gray_r')
#     axs[i].set_title(f'Cluster {i}')
#     axs[i].axis('off')
# plt.show()

# draw_fruits(fruits[km.labels_ == 0])
# draw_fruits(fruits[km.labels_ == 1])
# draw_fruits(fruits[km.labels_ == 2])
# draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
print(km.cluster_centers_.shape)
print(km.transform(fruits_2d[100:101])) # 101번째 과일 이미지에서 각 클러스터 중심까지의 거리
print(km.predict(fruits_2d[100:101]))
# draw_fruits(fruits[100:101])
# print(fruits[100:101].shape) # (1, 100, 100)
# print(fruits[100].shape) # (100, 100)

knumber = []
for k in range(2,7):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(fruits_2d)

    # Inertia 관성,무기력. 통계나 데이터 분석에서는 "움직이지 않는 정도",응집도
    # K-Means에서는 클러스터의 중심(Center)으로부터 그 그룹에 속한 데이터들이 얼마나 흩어지지 않고 똘똘 뭉쳐 있는지를 나타내는 척도
    knumber.append(km.inertia_) 

print(f'km.inertia_ array', knumber)
# 엘보우(Elbow) 차트
plt.plot(range(2,7), knumber)
# 1. 축 이름과 제목 추가
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia (Sum of squared distances)', fontsize=12)
plt.title('Elbow Method: Finding Optimal K', fontsize=14)
plt.show()


