import cv2, os, pathlib
from multiprocessing import Pool
import time


def cropping():
    filePath = '../original_test_image'
    #filePath = '../original_image'

    classifierPath = './cascade_classifier/'

    for classifier in os.listdir(classifierPath):
        faceCascade = cv2.CascadeClassifier(classifierPath + classifier)

        for directoryName in os.listdir(filePath):
            celebName = directoryName
            cascadePath = '../cropped_test_image/' + celebName
            #cascadePath = '../cropped_image/' + celebName
            pathlib.Path(cascadePath).mkdir(parents=True, exist_ok=True)
            for fileName in os.listdir(filePath + '/' + celebName):
                imageName = cv2.imread(filePath + '/' + celebName + '/' + fileName)
                gray = cv2.cvtColor(imageName, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        cropped = imageName[y:y+h, x:x+w]
                    os.chdir(cascadePath)

                    imageNameList = fileName.split('.')[0]
                    classifierNameList = classifier.split('.')[0]
                    fullName = imageNameList + '_' + classifierNameList + '.jpg'
                    cv2.imwrite(fullName, cropped)

                    os.chdir('../../tensorflow_test')


if __name__ == '__main__':
    p = Pool(8)
    startTime = int(time.time())
    p.apply(cropping())
    p.close()
    endTime = int(time.time())
    print('크로핑 완료, 소요 시간 : ', (endTime - startTime))
