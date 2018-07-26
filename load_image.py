# -*- coding: utf-8 -*-
import os
import glob

import numpy as np
import cv2

IMAGE_SIZE = 224
images = []
labels = []


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    resized_image = cv2.resize(image, (height, width))

    return resized_image


def traverse_dir(path):
    categories = os.listdir(path)
    x = []  # 이미지 데이터
    y = []  # 레이블 데이터

    for idx, cat in enumerate(categories):
        image_dir = path + "/" + cat
        files = glob.glob(image_dir + "/*.jpg")
        print('----', cat, '처리 중')

        for i, f in enumerate(files):
            img = read_image(f)
            x.append(img)
            y.append(idx)
    x = np.array(x)
    y = np.array(y, dtype=np.int)
    print(x)
    print(y)
    return x, y


def read_image(file_path):
    image = cv2.imread(file_path)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image


def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    print(labels)

    return images, labels


