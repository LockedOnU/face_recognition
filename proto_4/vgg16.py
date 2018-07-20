# -*- coding: utf-8 -*-
import numpy as np
import utils
import time
import os
import vgg16_trainable as vgg16
import tensorflow as tf

image_size = 224
root_dir = "../cropped_test_image/"
categories = os.listdir(root_dir)


def to_category(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.int)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def train_from_pre_trained():
    # n_class = len(categories)
    n_class = 49
    # y_train_one_hot = np.zeros((y_train.size, n_class), dtype=np.int)
    # # y_train_one_hot[np.arange(y_train.size), y_train] = 1
    # y_train_one_hot[np.arange(y_train.size), y_train-1] = 1
    # # y_train_one_hot = np_utils.to_categorical(y_train)

    data_2 = np.load("./data/celeb.npz")

    x_train = data_2['x_train']
    y_train = data_2['y_train']

    y_train = to_category(y_train)

    print(x_train)
    print(x_train.shape)
    print(y_train)
    print(type(y_train))
    print(len(y_train))

    len_files = len(y_train)

    batch_size = 32
    cost_all = []

    sess = tf.Session()

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, n_class])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg16.Vgg16('../vgg16.npy', n_class=n_class)
    # vgg = vgg16.Vgg16('data/test-save.npy', n_class=n_class)
    vgg.build(images, train_mode, int_image=True)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    # simple 1-step training
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    # for step in range(len_files * 1):
    for step in range(16250):
        batch_mask = np.random.choice(x_train.shape[0], batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]
        _, c = sess.run([train, cost], feed_dict={images: x_batch, true_out: y_batch, train_mode: True})

        if step % 1000 == 0:
            cost_all.append([step, c])
            print(step, c)
        np.savetxt("./data/cost.txt", cost_all)
    # test classification again, should have a higher probability about tiger

    # test accuracy 계산
    n_sample = 50
    n = int(x_train.shape[0] / n_sample)
    accuracy = 0
    for i in range(n):
        prob = sess.run(vgg.prob, feed_dict={images: x_train[i * 50:(i + 1) * 50], train_mode: False})
        prob = np.argmax(prob, axis=1)
        y_batch = y_train[i * 50:(i + 1) * 50]
        accuracy += np.sum(prob == y_batch) / float(n_sample)
    print("accuracy: ", accuracy / n)

    # plt.figure()
    # plt.title(category[y_batch[0]])
    # plt.imshow(x_batch[0]/255.0)
    # plt.show()
    # plt.close()
    # utils.print_prob(prob[0], './synset.txt')

    # test save
    vgg.save_npy(sess, './data/test-save.npy')


def predict_from_pre_trained():
    # n_class = len(categories)
    n_class = 49

    test_source = 1

    if test_source == 1:
        fileNames = ['./test_picture/0001.jpg', './test_picture/0002.jpg', './test_picture/0003.jpg',
                     './test_picture/0004.jpg','./test_picture/0005.jpg', './test_picture/0006.jpg', './test_picture/0007.jpg']
        label = [0,1,2,3,4,5,6]

        test_batch = None
        for i in range(len(fileNames)):
            img = utils.load_image(fileNames[i], img_size=image_size, float_flag=False).reshape(
                (1, image_size, image_size, 3))
            if test_batch is None:
                test_batch = img
            else:
                test_batch = np.concatenate((test_batch, img), 0)

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        train_mode = tf.placeholder(tf.bool)
        vgg = vgg16.Vgg16('./data/test-save.npy', trainable=False, n_class=n_class)
        with tf.name_scope("content_vgg"):
            vgg.build(images, train_mode, int_image=True)

        prob = sess.run(vgg.prob, feed_dict={train_mode: False, images: test_batch})
        # print(prob)
        prob_argmax = np.argmax(prob, axis=1)
        for i in range(min(100, test_batch.shape[0])):
            print("correct: ", categories[label[i]], "predict: ", categories[prob_argmax[i]],
                  label[i] == prob_argmax[i])

        print("acc: ", np.sum(label == prob_argmax) / float(test_batch.shape[0]))


if __name__ == "__main__":
    startTime = int(time.time())

    # train_from_pre_trained()
    predict_from_pre_trained()

    endTime = int(time.time())

    print('소요 시간 : ', (endTime - startTime))
