from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import numpy as np
from multiprocessing import Pool
import time
import os


def make_npy():
    root_dir = "../cropped_test_image/"
    # root_dir = "../cropped_image/"
    categories = os.listdir(root_dir)
    image_size = 50
    X = []  # 이미지 데이터
    Y = []  # 레이블 데이터
    for idx, cat in enumerate(categories):
        image_dir = root_dir + "/" + cat
        files = glob.glob(image_dir + "/*.jpg")
        print('----', cat, '처리 중')
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")  # 색상 모드 변경
            img = img.resize((image_size, image_size))  # 이미지 크기 변경
            data = np.asarray(img)
            X.append(data)
            Y.append(idx)
    X = np.array(X)
    Y = np.array(Y)
    # 학습 전용 데이터와 테스트 전용 데이터 분류하기 --- (※3)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=33)

    print(X_train.shape, y_train.shape)
    print(y_train)
    xy = (X_train, X_test, y_train, y_test)
    np.save("./data/celeb.npy", xy)
    print("ok,", len(Y))


if __name__ == '__main__':
    p = Pool(8)
    startTime = int(time.time())
    p.apply(make_npy)
    p.close()
    endTime = int(time.time())
    print('npy 파일 생성 완료, 소요 시간 : ', (endTime - startTime))