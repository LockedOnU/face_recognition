from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import os

root_dir = "../cropped_test_image/"
categories = os.listdir(root_dir)
nb_classes = len(categories)


def main():
    X_train, X_test, y_train, y_test = np.load("./data/celeb.npy")
    # 데이터 정규화하기
    X_train = X_train.astype('float') / 256
    X_test = X_test.astype('float') / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print(X_train.shape[1:])
    # 모델을 훈련하고 평가하기
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)


def model_train(X, y):
    model = vgg16_model(X.shape[1:])
    model.fit(X, y, batch_size=32, epochs=30)

    hdf5_file = "./data/celeb-model.hdf5"
    model.save_weights(hdf5_file)
    return model


def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])


def vgg16_model(in_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=in_shape))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))           # 112 * 112
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))           # 56 * 56
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))           # 28 * 28
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))           # 14 * 14
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))           # 7 * 7

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    main()
