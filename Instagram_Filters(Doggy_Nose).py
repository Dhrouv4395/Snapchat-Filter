import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
nose_img = cv2.imread('doggy_nose.png')
ears_img = cv2.imread('doggy_ears.png')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/Lav/Downloads/effect_1/shape_predictor_68_face_landmarks.dat')

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    
    for face in faces:
        landmarks1 = predictor(gray,face)
        bottom_ears = (landmarks1.part(18).x, landmarks1.part(18).y)
        left_ears = (landmarks1.part(17).x,landmarks1.part(17).y)
        right_ears = (landmarks1.part(26).x,landmarks1.part(26).y)
        center_ears = (landmarks1.part(27).x, landmarks1.part(27).y)
        ears_width = int(hypot(left_ears[0] - right_ears[0],left_ears[1] - right_ears[1]) * 1.5)
        ears_height = int(ears_width * 0.77)
        cv2.rectangle(frame,(left_ears),(right_ears),(255,0,0),2)
        doggy_ears = cv2.resize(ears_img,(ears_width,ears_height))
        doggy_ears_gray = cv2.cvtColor(doggy_ears,cv2.COLOR_BGR2GRAY)
        _,ears_mask = cv2.threshold(doggy_ears_gray,25,255,cv2.THRESH_BINARY_INV)

        top_left = (int(center_ears[0]-ears_width/2),
                             int(center_ears[1]-ears_height/2))
        bottom_right = (int(center_ears[0]+ears_width/2),
                    int(center_ears[1]+ears_height/2))
        ears_area = frame[top_left[1]:top_left[1]+ears_height,
                          top_left[0]:top_left[0]+ears_width]
        ears_area_no_ears = cv2.bitwise_and(ears_area,ears_area, mask = ears_mask)
        cv2.imshow('Ears',ears_area_no_ears)
        
        #cv2.imshow('Ears',doggy_ears)        
        #cv2.imshow('Ears_Frame',ears_mask)
        
        
    for face in faces:
        landmarks = predictor(gray,face)
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        nose_width = int(hypot(left_nose[0]-right_nose[0],
                           left_nose[1]-right_nose[1])*1.25)
        nose_height = int(nose_width * 0.77)

        #new nose position
        top_left = (int(center_nose[0]-nose_width/2),
                             int(center_nose[1]-nose_height/2))
        bottom_right = (int(center_nose[0]+nose_width/2),
                    int(center_nose[1]+nose_height/2))

        nose_pig = cv2.resize(nose_img,(nose_width,nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig,cv2.COLOR_BGR2GRAY)
        _,nose_mask = cv2.threshold(nose_pig_gray,25,255,cv2.THRESH_BINARY_INV)
        
        nose_area = frame[top_left[1]:top_left[1]+nose_height,
                          top_left[0]:top_left[0]+nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area,nose_area, mask = nose_mask)
        cv2.imshow('Nose',nose_area_no_nose)
        final_nose = cv2.add(nose_area_no_nose,nose_pig)
        frame[top_left[1]:top_left[1]+nose_height,
                          top_left[0]:top_left[0]+nose_width]=final_nose
        
        
    cv2.imshow('Frame_Nose',frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
