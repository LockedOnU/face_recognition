from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
import os

root_dir = "../cropped_test_image/"
# root_dir = "../cropped_image/"
categories = os.listdir(root_dir)
nb_classes = len(categories)
image_size = 50


def main():
    X_train, X_test, y_train, y_test = np.load("./data/celeb.npy")
    # 데이터 정규화하기
    X_train = X_train.astype("float") / 256
    X_test  = X_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test  = np_utils.to_categorical(y_test, nb_classes)
    # 모델을 훈련하고 평가하기
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)


def build_model(in_shape):
    model = Sequential()
    # Convolution
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=in_shape))
    model.add(Activation('relu'))

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Convolution Layer
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())

    # Full connection
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # Compiling CNN
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def model_train(X, y):
    model = build_model(X.shape[1:])
    model.fit(X, y, batch_size=32, epochs=1)

    hdf5_file = "./data/celeb-model.hdf5"
    model.save_weights(hdf5_file)
    return model


def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])


if __name__ == "__main__":
    main()
