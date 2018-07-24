from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
import numpy as np
import os

root_dir = "../cropped_test_image/"
categories = os.listdir(root_dir)
nb_classes = len(categories)
image_size = 224


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = np.load("./data/celeb.npy")

    print(K.image_dim_ordering())
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, image_size, image_size)
        X_valid = X_valid.reshape(X_valid.shape[0], 3, image_size, image_size)
        X_test = X_test.reshape(X_test.shape[0], 3, image_size, image_size)
    else:
        X_train = X_train.reshape(X_train.shape[0], image_size, image_size, 3)
        X_valid = X_valid.reshape(X_valid.shape[0], image_size, image_size, 3)
        X_test = X_test.reshape(X_test.shape[0], image_size, image_size, 3)


    # 데이터 정규화하기
    X_train = X_train.astype('float32') / 255
    X_valid = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_valid = np_utils.to_categorical(y_valid, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    print(X_train.shape[1:])
    # 모델을 훈련하고 평가하기
    model = model_train(X_train, y_train, X_valid, y_valid)
    model_eval(model, X_test, y_test)


def model_train(X, y, X_valid, y_valid):
    model = vgg16_model(X.shape[1:])
    model.fit(X, y, batch_size=32, nb_epoch=100, shuffle=True)

    hdf5_file = "./data/celeb-model.hdf5"
    model.save_weights(hdf5_file)
    return model


def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])


def vgg16_model(in_shape):
    model = Sequential()

    # Convolution
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Convolution Layer
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Convolution Layer
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Convolution Layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Convolution Layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Flattening
    model.add(Flatten())
    # Full connection
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


if __name__ == "__main__":
    main()
