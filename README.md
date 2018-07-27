# face_recognition (vgg16 모델을 사용한 얼굴 인식 프로그램)

## 0. Pref

### 0.1. CNN 모델 선택 (VGG16)



## 1. Face-Predict

### 1.1. image_scrapper.py
학습할 이미지 데이터들을 웹에서 가져오는 작업이다.

### 1.2. crop_image.py
저장한 Original Image 들을 하나씩 읽고 얼굴이 있는지 판별한 후 있다면 얼굴 영역만 Crop하여 정사각형의 Image를 생성하는 작업이다.

### 1.3. make_dataSet.py
Cropped Image들을 배열화하고 Data-set 파일을 생성하는 작업이다.

### 1.4. train.py
Data-set을 학습하는 작업이다.

### 1.5. predict_face.py
Data-set을 학습한 가중치를 통해 시스템 인자로 넘겨준 이미지 속 인물이 누군지 예측한다.

## 2. Realtime Face-Predict with WebCam

### 2.1. crop_image_webcam.py
웹캠이 동작하면서 최초 동작 시 프레임 내에 얼굴이 있는지 판별한 후 있다면 얼굴 영역을 Tracker(KCF)로 추적을 시작하고 일정 프레임마다(Default : 10Frame) 얼굴 영역을 Crop한 정사각형의 Image를 생성하는 작업이다.

### 2.2. load_image.py
이미지들을 읽어와 배열데이터를 생성하는 작업이다.

### 2.3. train_webcam.py
numpy 배열 데이터를 학습하는 작업이다. (Data-set 파일을 생성하는 과정은 생략하고 바로 학습하도록 설계하였다.)

### 2.4. predict_face_webcam.py
최초 동작시 학습한 가중치 모델을 로드한 후 웹캠이 촬영하는 프레임 내에 얼굴이 있는지 판별한 후 있다면 얼굴 영역을 Tracker(KCF)로 추적을 시작하고 일정 프레임마다(Default : 10Frame) 추적 중인 인물이 누군지 예측한다.

## ※ 참고
### 1. 참고 문헌
  - Wes Mckinney. (2013). 파이썬 라이브러리를 활용한 데이터 분석 (김영근, 옮김). 한빛미디어. (원서출판 2011).
  - W. Saito Goki. (2017). 밑바닥부터 시작하는 딥러닝: 파이썬으로 익히는 딥러닝 이론과 구현 (이복연, 옮김). 한빛미디어. (원서출판 2016).
  - Kujira Hikouzukue. (2017). 파이썬을 이용한 머신러닝, 딥러닝 실전 개발 입문 (윤인성, 옮김). 위키북스. (원서출판 2016).
  
### 2. 참고 사이트
  - __마크다운 사용법 :__ <https://gist.github.com/ihoneymon/652be052a0727ad59601>
  - __Github에 100MB 이상의 파일을 올리는 방법 :__ <https://medium.com/@stargt/github%EC%97%90-100mb-%EC%9D%B4%EC%83%81%EC%9D%98-%ED%8C%8C%EC%9D%BC%EC%9D%84-%EC%98%AC%EB%A6%AC%EB%8A%94-%EB%B0%A9%EB%B2%95-9d9e6e3b94ef>
