import cv2
import numpy as np
from tensorflow import keras

(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()
print(train_X.shape, train_y.shape) # (60000, 28, 28) (60000,)
print(test_X.shape, test_y.shape) # (10000, 28, 28) (10000,)

# 28x28 이미지를 20x20으로 리사이즈
train_X = np.array([cv2.resize(img, (20, 20)) for img in train_X])
test_X = np.array([cv2.resize(img, (20, 20)) for img in test_X])
print(train_X.shape, train_y.shape) # (60000, 20, 20) (60000,)
print(test_X.shape, test_y.shape) # (10000, 20, 20) (10000,)

# CNN 모델에 입력하기 위해 4D 텐서로 변환하고 정규화
# (60000, 20, 20) -> (60000, 20, 20, 1)
train_X = train_X.reshape(-1, 20, 20,1).astype(np.float32) / 255.0
test_X = test_X.reshape(-1, 20, 20,1).astype(np.float32) / 255.0
print(train_X.shape, train_y.shape) # (60000, 20, 20, 1) (60000,)
print(test_X.shape, test_y.shape) # (10000, 20, 20, 1) (10000,)

# CNN 모델
model = keras.Sequential([
    keras.Input(shape=(20, 20, 1)),
    keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
model.fit(train_X, train_y, epochs=20, validation_split=0.2, callbacks=[early_stop])

loss, acc = model.evaluate(test_X, test_y)
print(f"테스트 정확도: {acc:.4f}") # 0.9906
model.save("data/mnist_cnn_model.keras")