import cv2, os, pathlib

filePath = '../CASIA-WebFace/CASIA-WebFace'
faceCascade = cv2.CascadeClassifier('../opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

for directoryName in os.listdir(filePath):
    celebName = directoryName
    cascadePath = '../cascadedImage/' + celebName
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
                # cv2.rectangle(imageName, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped = imageName[y:y+h, x:x+w]
            os.chdir(cascadePath)
            # cv2.imshow('Faces found', imageName)
            cv2.imwrite(fileName, cropped)
            os.chdir('../../test')
            # cv2.waitKey(500)
            # cv2.destroyAllWindows()
