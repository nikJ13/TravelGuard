import cv2
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.io
import io

from scipy.misc import imsave


from skimage import data, img_as_float
from skimage import exposure


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #print(type(frame))
    
    cv2.imwrite('image.jpg', frame)
    img = data.imread('image.jpg')
    #print(type(img))
    #imsave('img.jpg',img)
    frame1 = exposure.equalize_hist(frame)
    imsave('test2.jpg',frame1)
    img1=cv2.imread('test2.jpg')
    
    faces = faceCascade.detectMultiScale(
        img1,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', img1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
















