from fastapi import FastAPI
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from pydantic import BaseModel # uvcorn 설치 할 때 같이 설치됨

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7,
               31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5,
               34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0,
               38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 10.5, 10.6, 11.0, 11.2,
               11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0,
               475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0,
               575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0,
               920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 7.5, 7.0, 9.7, 9.8,
               8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

app = FastAPI()
fish_data = list(zip(fish_length, fish_weight))
fish_target = [1] * 35 + [0] * 14

# 1. 데이터를 넘파이 배열로 변환
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
print(input_arr.shape, target_arr.shape)

# 2. 인덱스를 만들고 섞기
np.random.seed(42) # 실행할 때마다 결과가 같게 하기 위해 설정
index = np.arange(49)
np.random.shuffle(index)
X = input_arr[index]
y = target_arr[index]

# 3. 섞인 인덱스로 데이터 나누기
X_train = X[:35]
y_train = y[:35]
X_test = X[35:]
y_test = y[35:]

kn = KNeighborsClassifier()
kn.fit(X_train, y_train)

# new1 = [[30,600]]
# prediction = kn.predict(new1)
# print(prediction)
# new1 = [[9.8,6.7]]
# prediction = kn.predict(new1)
# print(prediction)

@app.get("/")
def root():
    return {"message":"Hello FastAPI!111"}


class Fish(BaseModel):
    length:float
    weight:float

@app.post("/pred")
def predict_fish(fish:Fish):
    data = [[fish.length, fish.weight]]
    pred = kn.predict(data)[0]
    result = "도미" if pred == 1 else "빙어"
    return { "prediction":result, "class" : int(pred) }
