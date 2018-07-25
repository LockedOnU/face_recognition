from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as k
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

root_dir = "../cropped_test_image/"
categories = os.listdir(root_dir)
nb_classes = len(categories)
image_size = 224


def main():
    x_train, x_valid, x_test, y_train, y_valid, y_test = np.load("./data/celeb.npy")

    print(k.image_dim_ordering())
    if k.image_dim_ordering() == 'th':
        x_train = x_train.reshape(x_train.shape[0], 3, image_size, image_size)
        x_valid = x_valid.reshape(x_valid.shape[0], 3, image_size, image_size)
        x_test = x_test.reshape(x_test.shape[0], 3, image_size, image_size)
    else:
        x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 3)
        x_valid = x_valid.reshape(x_valid.shape[0], image_size, image_size, 3)
        x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 3)

    # 데이터 정규화하기
    x_train = x_train.astype('float32') / 255
    x_valid = x_valid.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_valid = np_utils.to_categorical(y_valid, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    print(x_train.shape[1:])
    # 모델을 훈련하고 평가하기
    model = model_train(x_train, y_train, x_valid, y_valid)
    model_eval(model, x_test, y_test)


def model_train(X, y, x_valid, y_valid):
    model = vgg16_model(X.shape[1:])

    data_gen = ImageDataGenerator(
        featurewise_center=False,             # set input mean to 0 over the dataset
        samplewise_center=False,              # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,   # divide each input by its std
        zca_whitening=False,                  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,                # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,               # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    data_gen.fit(X)

    model.fit_generator(data_gen.flow(X, y, batch_size=32), samples_per_epoch=X.shape[0], nb_epoch=100,
                        validation_data=(x_valid, y_valid))

    model.fit(X, y, batch_size=32, nb_epoch=100)

    hdf5_file = "./data/celeb-model.hdf5"
    model.save_weights(hdf5_file)
    return model


def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])


def vgg16_model(in_shape):
    model = Sequential()

    # # Convolution
    # model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    # model.add(Activation('relu'))
    # model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # # Pooling
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # # Convolution Layer
    # model.add(Conv2D(128, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(128, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # # Pooling
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # # Convolution Layer
    # model.add(Conv2D(256, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(256, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(256, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # # Pooling
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # # Convolution Layer
    # model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # # Pooling
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # # Convolution Layer
    # model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # # Pooling
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # # Flattening
    # model.add(Flatten())
    # # Full connection
    # model.add(Dense(4096))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))

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

    model.summary()

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


if __name__ == "__main__":
    main()
