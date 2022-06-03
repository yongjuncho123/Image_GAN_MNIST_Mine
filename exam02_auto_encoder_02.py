# GAN의 기본은 Auto_encoder

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import  mnist

# 모델을 더 키워보자
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img) # 인풋 넣기
encoded = Dense(64, activation='relu')(encoded) #앞의 encoded를 인풋으로
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='relu')(decoded)

# Auto_encoder
autoencoder = Model(input_img, decoded) # 모델(인풋값, 출력값)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

(x_train, _), (x_test, _)  = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
flatted_x_train = x_train.reshape(-1, 28*28)
flatted_x_test = x_test.reshape(-1, 28*28)
print(flatted_x_train.shape)
print(flatted_x_test.shape)




# 학습                     # 입력된값으 압축되서 그대로 나와야해서 입출력값이 같다
fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train,
                           epochs=50, batch_size=256,
                           validation_data=(flatted_x_test, flatted_x_test))



decoded_img = autoencoder.predict(flatted_x_test[:10])

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i+1) # 첫줄
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i+1+n) # 둘째줄(2행 10열이니 2번째줄의 11번째부터)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
