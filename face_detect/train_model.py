import os
import cv2
import numpy as np 

face_cascade = cv2.CascadeClassifier('/Users/zhancheng-ibm/anaconda2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('/Users/zhancheng-ibm/anaconda2/share/OpenCV/haarcascades/haarcascade_eye.xml')

face_recognizer = cv2.face.createLBPHFaceRecognizer()
images = []
labels = []
os.chdir('./face_detect/train_mike')
for idx in range(1,113):
    img_location = "/Users/zhancheng-ibm/Desktop/opencv-projects/face_detect/train_mike/img-"+str(idx)+".jpg"
    print(img_location)
    image = cv2.imread(img_location,0)
    if image is not None:
        images.append(image)
        labels.append(1)
face_recognizer.train(images,np.array(labels))

for idx in range(1,151):
    img_location = "/Users/zhancheng-ibm/Desktop/opencv-projects/face_detect/train_matt/img-"+str(idx)+".jpg"
    print(img_location)
    image = cv2.imread(img_location,0)
    if image is not None:
        images.append(image)
        labels.append(2)
face_recognizer.train(images,np.array(labels))

for idx in range(1,130):
    img_location = "/Users/zhancheng-ibm/Desktop/opencv-projects/face_detect/train_york/img-"+str(idx)+".jpg"
    print(img_location)
    image = cv2.imread(img_location,0)
    if image is not None:
        images.append(image)
        labels.append(3)
face_recognizer.train(images,np.array(labels))

print("Face Recongnition Trainning done")
face_recognizer.save("faceRecong.xml")

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
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
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        label = face_recognizer.predict(gray[y:y+h, x:x+w])
        if label == 1:
            cv2.putText(frame, 'Mike', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        if label == 2:
            cv2.putText(frame, 'MATTTTTT', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        if label == 3:
            cv2.putText(frame, 'York', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        

    cv2.imshow("Frame", frame)