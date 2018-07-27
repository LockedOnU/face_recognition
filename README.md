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

## - 참조
### - 참고한 사이트
