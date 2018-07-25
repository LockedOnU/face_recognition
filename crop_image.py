import cv2
import os
import pathlib


def cropping():
    file_path = '../original_test_image'

    classifier_path = './cascade_classifier/'

    # for classifier in os.listdir(classifier_path):
    #     face_cascade = cv2.CascadeClassifier(classifier_path + classifier)
    face_cascade = cv2.CascadeClassifier(classifier_path + 'haarcascade_frontalface_default.xml')
    for directoryName in os.listdir(file_path):
        celeb_name = directoryName
        cascade_path = '../cropped_test_image/' + celeb_name
        pathlib.Path(cascade_path).mkdir(parents=True, exist_ok=True)
        for fileName in os.listdir(file_path + '/' + celeb_name):
            image_name = cv2.imread(file_path + '/' + celeb_name + '/' + fileName)
            gray = cv2.cvtColor(image_name, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cropped = image_name[y:y+h, x:x+w]
                os.chdir(cascade_path)

                # imageNameList = fileName.split('.')[0]
                # classifierNameList = classifier.split('.')[0]
                # fullName = imageNameList + '_' + classifierNameList + '.jpg'
                cv2.imwrite(fileName, cropped)

                os.chdir('../../tensorflow_test')


cropping()
