import os
import cv2
# load the cascade
# cv2 also designed to solve computer vision problem
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# to capture video from web cam
# haar cascade is a object detection algorithim 
classifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

dirFace = 'cropped_face'
# Create if there is no cropped face directory
if not os.path.exists(dirFace):
    os.mkdir(dirFace)
    print("Directory " , dirFace ,  " Created ")
else:    
    print("Directory " , dirFace ,  " has found.")

webcam = cv2.VideoCapture(0) # Camera 0 according to USB port
# video = cv2.VideoCapture(r"use full windows path") # video path
cap = cv2.VideoCapture(0)
# cap is a variabale
# to use a video file as input 
# cv2.videocapture('filename.mp4')
while True :
    # read the frame
    _, img = cap.read()
    # convert to gray scale
    #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)= this expression is used to convert image from one color space to another
    # default color format in opencv is often refered as RGB but it actually uses BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect the face 
    #detectMultiScale = It takes 3 common arguments — the input image, scaleFactor(to convert big image into samll so that algorithm can detect), and minNeighbours
    (f, im) = webcam.read() # f returns only True, False according to video access
    # (f, im) = video.read() # video 

    if f != True:
       break

    # im=cv2.flip(im,1,0) #if you would like to give mirror effect
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # draw the rectangle around each face
    # (img, (x,y), (x+w, y+h), (255,0,0), 2) = img , pt1 , pt2 , brightness or rectangle color , thickness
    # detectfaces 
    faces = classifier.detectMultiScale(
        im, # stream 
        scaleFactor=1.10, # change these parameters to improve your video processing performance
        minNeighbors=20, 
        minSize=(30, 30) # min image detection size
        ) 
    # Draw rectangles around each face
    for (x, y, w, h) in faces:

        cv2.rectangle(im, (x, y), (x + w, y + h),(0,0,255),thickness=2)
        # saving faces according to detected coordinates 
        sub_face = im[y:y+h, x:x+w]
        FaceFileName = "cropped_face/face_" + str(y+x) + ".jpg" # folder path and random name image
        cv2.imwrite(FaceFileName, sub_face)    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        #display
    cv2.imshow('img' , img)
    # Video Window
    cv2.imshow('Video Stream',im)
    if cv2.waitKey(1) :
          break
cap.release()   
cv2.destroyAllWindows()             