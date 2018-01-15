import numpy as np
import cv2 
import os
# Set the classifier to the pre-trained front face 
face_cascade = cv2.CascadeClassifier('/Users/zhancheng-ibm/anaconda2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('/Users/zhancheng-ibm/anaconda2/share/OpenCV/haarcascades/haarcascade_eye.xml')
# Get the camera from opencv
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
imgIdx=0
#os.chdir('train_mike')
print(os.curdir)
os.chdir('./face_detect/train_york')
while True:
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
	    break
    (grabbed, frame)=camera.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30))
    for (x,y,w,h) in faces :
        cv2.imwrite("img-"+str(imgIdx)+".jpg",frame[y:(y+h),x:(x+w)] )
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    

    cv2.imshow("Frame", frame)
    
    imgIdx=imgIdx+1
 
	