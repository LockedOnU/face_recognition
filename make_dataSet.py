from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import numpy as np
from multiprocessing import Pool
import time
import os
import random


def make_npy():
    root_dir = "../cropped_test_image/"
    # root_dir = "../cropped_image/"
    categories = os.listdir(root_dir)
    image_size = 224
    x = []  # 이미지 데이터
    y = []  # 레이블 데이터
    for idx, cat in enumerate(categories):
        image_dir = root_dir + "/" + cat
        files = glob.glob(image_dir + "/*.jpg")
        print('----', cat, '처리 중')
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")  # 색상 모드 변경
            img = img.resize((image_size, image_size))  # 이미지 크기 변경
            data = np.asarray(img)
            x.append(data)
            y.append(idx)
    x = np.array(x)
    y = np.array(y, dtype=np.int)

    # 학습 전용 데이터와 테스트 전용 데이터 분류하기 --- (※3)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random.randint(0, 100))
    X_valid, X_test, y_valid, y_test = train_test_split(x, y, test_size=0.5, random_state=random.randint(0, 100))
    xy = (X_train, X_valid, X_test, y_train, y_valid, y_test)
    np.save("./data/celeb", xy)

    # np.savez_compressed("./data/celeb", x_train=x, y_train=y)
    print("ok,", len(y))


if __name__ == '__main__':
    # p = Pool(8)
    startTime = int(time.time())
    # p.apply(make_npy)
    # p.close()
    make_npy()
    endTime = int(time.time())
    print('npy 파일 생성 완료, 소요 시간 : ', (endTime - startTime))
