import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# 5개의 세트 데이터 (X), 각각 8개의 feature
X = np.array([
    [135, 150, 7, 60, 10, 90, 1, 0],
    [140, 155, 8, 60, 10, 90, 2, 0],
    [145, 160, 9, 60, 10, 90, 3, 0],
    [150, 165, 9, 60, 10, 90, 4, 0],
    [155, 170, 10, 60, 10, 90, 5, 0]
])

# 정답값 Y: 다음 세트에서 강도를 어떻게 할지
# 0: 강도 낮춤, 1: 유지, 2: 증가
y = np.array([1, 1, 1, 0, 0])


model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(8,)))  # feature 개수
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 분류: 3가지 클래스

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, verbose=1)

# 예측할 새로운 세트 데이터
new_set = np.array([[158, 172, 9, 60, 10, 90, 6, 0]])
pred = model.predict(new_set)

print("각 클래스 확률:", pred)
print("예측된 클래스:", np.argmax(pred))
