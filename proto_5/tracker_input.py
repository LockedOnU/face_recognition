# -*- coding: utf-8 -*-
import os
import glob

import numpy as np
import cv2

IMAGE_SIZE = 224


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    # def get_padding_size(image):
    #     h, w, _ = image.shape
    #     longest_edge = max(h, w)
    #     top, bottom, left, right = (0, 0, 0, 0)
    #     if h < longest_edge:
    #         dh = longest_edge - h
    #         top = dh // 2
    #         bottom = dh - top
    #     elif w < longest_edge:
    #         dw = longest_edge - w
    #         left = dw // 2
    #         right = dw - left
    #     else:
    #         pass
    #     return top, bottom, left, right
    #
    # top, bottom, left, right = get_padding_size(image)
    # BLACK = [0, 0, 0]
    # constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(image, (height, width))

    return resized_image


images = []
labels = []


def traverse_dir(path):
    # for file_or_dir in os.listdir(path):
    #     abs_path = os.path.abspath(os.path.join(path, file_or_dir))         # 절대 경로 내에 디렉터리만 추출
    #     print(abs_path)
    #     if os.path.isdir(abs_path):  # dir
    #         traverse_dir(abs_path)                                          # 디렉터리 있으면 다시 함수 호출
    #     else:                        # file
    #         if file_or_dir.endswith('.jpg'):
    #             image = read_image(abs_path)
    #             images.append(image)
    #             labels.append(path)

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
    # labels = np.array([1 for label in labels])
    print(labels)

    return images, labels


