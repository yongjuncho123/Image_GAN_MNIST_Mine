# GAN의 기본은 Auto_encoder

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import  mnist

# 모델을 CNN으로
input_img = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x) # 맥스풀은 이미지 사이즈를 작게해주는
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x) # 28 28 픽셀을 봐야하는데 패딩을 주니 32가 나와서 패딩을 빼 14로 맞춰줌
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()



# Auto_encoder
# autoencoder = Model(input_img, decoded) # 모델(인풋값, 출력값)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.summary()

(x_train, _), (x_test, _)  = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
conv_x_train = x_train.reshape(-1, 28, 28, 1)
conv_x_test = x_test.reshape(-1, 28, 28, 1)
print(conv_x_train.shape)
print(conv_x_test.shape)




# 학습                     # 입력된값으 압축되서 그대로 나와야해서 입출력값이 같다
fit_hist = autoencoder.fit(conv_x_train, conv_x_train,
                           epochs=50, batch_size=256,
                           validation_data=(conv_x_test, conv_x_test))



decoded_img = autoencoder.predict(conv_x_test[:10])

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
