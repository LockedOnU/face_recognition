# -*- coding: utf-8 -*-
from __future__ import print_function
import random

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Conv2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as k

from load_image import extract_data, resize_with_pad, IMAGE_SIZE


class DataSet(object):

    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None

    def read(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, nb_classes=63):
        images, labels = extract_data('../cropped_test_image/')
        labels = np.reshape(labels, [-1])
        # numpy.reshape
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3,
                                                            random_state=random.randint(0, 100))
        x_valid, x_test, y_valid, y_test = train_test_split(images, labels, test_size=0.5,
                                                            random_state=random.randint(0, 100))
        print(k.image_dim_ordering())
        if k.image_dim_ordering() == 'th':
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            x_valid = x_valid.reshape(x_valid.shape[0], 3, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

        # the data, shuffled and split between train and test sets
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_valid.shape[0], 'valid samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_valid = np_utils.to_categorical(y_valid, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)

        x_train = x_train.astype('float32')
        x_valid = x_valid.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_valid /= 255
        x_test /= 255

        self.X_train = x_train
        self.X_valid = x_valid
        self.X_test = x_test
        self.Y_train = y_train
        self.Y_valid = y_valid
        self.Y_test = y_test


class Model(object):

    FILE_PATH = './data/model.h5'

    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=63):
        self.model = Sequential()

        # Convolution
        self.model.add(Conv2D(64, (3, 3), padding='same', input_shape=dataset.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        # Pooling
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Convolution Layer
        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        # Pooling
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Convolution Layer
        self.model.add(Conv2D(256, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(256, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(256, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        # Pooling
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Convolution Layer
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        # Pooling
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Convolution Layer
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        # Pooling
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Flattening
        self.model.add(Flatten())
        # Full connection
        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train(self, dataset, batch_size=32, nb_epoch=40, data_augmentation=True):
        # let's train the model using SGD + momentum (how original).
        #
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(dataset.X_train, dataset.Y_train,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           # validation_data=(dataset.X_valid, dataset.Y_valid),
                           shuffle=True)
        else:
            print('Using real-time data augmentation.')

            # this will do preprocessing and realtime data augmentation
            data_gen = ImageDataGenerator(
                rotation_range=20,                     # randomly rotate images in the range (degrees, 0 to 180)
            )

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            data_gen.fit(dataset.X_train)

            # fit the model on the batches generated by data_gen.flow()
            self.model.fit_generator(data_gen.flow(dataset.X_train, dataset.Y_train, batch_size=batch_size),
                                     samples_per_epoch=dataset.X_train.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.X_valid, dataset.Y_valid))

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)
        # self.model = np.load(file_path)

    def predict(self, image):
        if k.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_with_pad(image)
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        elif k.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_with_pad(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        image = image.astype('float32')
        image /= 255
        result = self.model.predict_classes(image)

        return result[0]

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))


if __name__ == '__main__':
    data_set = DataSet()
    data_set.read()

    model = Model()
    model.build_model(data_set)
    model.train(data_set, nb_epoch=200)
    model.save()

    model = Model()
    model.load()
    model.evaluate(data_set)
