from imutils import face_utils
import dlib
import cv2
import numpy as np

p = 'predictor/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

image = cv2.imread('images/2-people.jpeg')
image = cv2.resize(image, (400, 400))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 0)

for (i, rect) in enumerate(rects):
    print(i, rect)
    print(rect)
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    print(shape)

    for (i, (x, y)) in enumerate(shape):
        cv2.putText(image, str(i), (x, y),  cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 2)

        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

cv2.imshow('Output', image)
cv2.waitKey(0)