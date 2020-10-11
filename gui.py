from __future__ import division
import time
from tkinter import *
import cv2
import face_recognition
from PIL import ImageTk,Image
import numpy as geek
import tkinter.font as font
global window
window = Tk()
window.title("EMOTION RECOGNITION MODEL")
window.config(bg='grey')

def work():
    import time
    from skimage import exposure
    import threading
    #from scipy.misc import imageio.imwrite
    import imageio
    from urllib.request import urlopen
    import cv2
    import numpy as np
    import tensorflow as tf
    #url='http://192.168.0.101:8080/shot.jpg'

    def violence(s):
        import boto3
        from botocore.client import Config

        ACCESS_KEY_ID = 'AKIA3567LTQY7YZCF5M7'
        ACCESS_SECRET_KEY = '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'
        BUCKET_NAME = 'newintelbucket'
        data = open(s, 'rb')

        s3 = boto3.resource(
            's3',
            aws_access_key_id=ACCESS_KEY_ID,
            aws_secret_access_key=ACCESS_SECRET_KEY,
            config=Config(signature_version='s3v4')
        )
        s3.Bucket(BUCKET_NAME).put_object(Key=s, Body=data)


        def moderate_image(photo, bucket):

            client=boto3.client('rekognition')

            response = client.detect_moderation_labels(Image={'S3Object':{'Bucket':bucket,'Name':photo}})
            #print(response)
            print('Detected labels for ' + photo)    
            for label in response['ModerationLabels']:
                print (label['Name'] + ' : ' + str(label['Confidence']))
            #print (label['ParentName'])
            return len(response['ModerationLabels'])
        
        # under violence function
        
        photo=s
        bucket='newintelbucket'
        label_count=moderate_image(photo, bucket) # calling 'moderate_image' function to compare the video from the s3 bucket images and give output of explicit content or not
        print("Labels detected: " + str(label_count))
        if str(label_count)=='0':
            print('No violence detected')

    def show_webcam(vs) :    
        import datetime
        import numpy as np
        import pandas as pd
        from time import time
        from time import sleep
        import re
        import os
        import math
        import argparse
        from collections import OrderedDict
        from scipy.ndimage import zoom
        from scipy.spatial import distance
        import imutils
        from scipy import ndimage
        import dlib
        from tensorflow.keras.models import load_model
        from imutils import face_utils
        import requests
        global shape_x
        global shape_y
        global input_shape
        global nClasses
        from imutils import face_utils
        from threading import Thread
        import numpy as np
        import playsound
        import argparse
        import imutils
        import dlib
        import simpleaudio as sa

        shape_x = 48
        shape_y = 48
        input_shape = (shape_x, shape_y, 1)
        nClasses = 7
        thresh = 0.25
        frame_check = 20

# =============================================================================
#         def prepare(filepath):
#             IMG_SIZE=150
#             img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
#             new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
#             return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)
# =============================================================================

        def sound_alarm():
            filename = 'alarm.wav'
            wave_obj = sa.WaveObject.from_wave_file(filename)
            play_obj = wave_obj.play()

        def eye_aspect_ratio(eye):
            A = distance.euclidean(eye[1], eye[5])
            B = distance.euclidean(eye[2], eye[4])
            C = distance.euclidean(eye[0], eye[3])
            ear = (A + B) / (2.0 * C)
            return ear

# =============================================================================
#         def detect_face(frame):
#             
#             # cascPath = 'face_landmarks.dat'
#             cascPath = 'shape_predictor_68_face_landmarks.dat'
#             faceCascade = cv2.CascadeClassifier(cascPath)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             detected_faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6,
#                                                           minSize=(shape_x, shape_y),
#                                                           flags=cv2.CASCADE_SCALE_IMAGE)
#             coord = []
#                                                           
#             for x, y, w, h in detected_faces :
#                 if w > 100 :
#                     sub_img=frame[y:y+h,x:x+w]
#                     cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,255),1)
#                     coord.append([x,y,w,h])
# 
#             return gray, detected_faces, coord
# =============================================================================

# =============================================================================
#         def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
#             gray = faces[0]
#             detected_face = faces[1]
#             
#             new_face = []
#             
#             for det in detected_face :
#                 
#                 x, y, w, h = det
#                 
#                 horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
#                 vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
#                 extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
#                 new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0],shape_y / extracted_face.shape[1]))
#                 new_extracted_face = new_extracted_face.astype(np.float32)
#                 new_extracted_face /= float(new_extracted_face.max())
#                 new_face.append(new_extracted_face)
#             
#             return new_face
# =============================================================================
        
        # still under show_webcam function
        
        EYE_AR_THRESH = 0.24        
        EYE_AR_CONSEC_FRAMES = 6
        COUNTER = 0
        ALARM_ON = False
        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]    
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
        (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        model = load_model('video.h5')
        face_detect = dlib.get_frontal_face_detector()
        # predictor_landmarks  = dlib.shape_predictor("face_landmarks.dat")
        predictor_landmarks  = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        count=0
        
        while True:
            
            while True:
                # Capture frame-by-frame
                '''imageResp=urlopen(url)
                imgNp=np.array(bytearray(imageResp.read()),dtype=np.uint8)
                frame1=cv2.imdecode(imgNp,-1)'''
                ret, frame1 = vs.read()
                count=count+1
                #frame = imutils.resize(frame1, width=450)
                if int(str(datetime.datetime.now())[11]+str(datetime.datetime.now())[12])>=18:
                    frame = exposure.equalize_hist(frame1)
                    imageio.imwrite('test2.jpg',frame1)        
                else:
                    cv2.imwrite('test2.jpg',frame1)

                img1=cv2.imread('test2.jpg')                
                gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                if count==1 or count==300:
                    violence('test2.jpg')
                    count=2
                face_index = 0
                rects = face_detect(gray, 1)
                
                for (i, rect) in enumerate(rects):
                    try:
                        shape = predictor_landmarks(gray, rect)
                        shape = face_utils.shape_to_np(shape)
                        (x, y, w, h) = face_utils.rect_to_bb(rect)
                        face = gray[y:y+h,x:x+w]
                        face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))
                        face = face.astype(np.float32)
                        face /= float(face.max())
                        face = np.reshape(face.flatten(), (1, 48, 48, 1))
                        prediction = model.predict(face)
                        prediction_result = np.argmax(prediction)
                        #print("Angry : " + str(round(prediction[0][0],3)))
                        #print("Disgust : " + str(round(prediction[0][1],3)))
                        '''print("Fear : " + str(round(prediction[0][2],3)))
                          print("Happy : " + str(round(prediction[0][3],3)))
                          print("Sad : " + str(round(prediction[0][4],3)))
                          print("Surprise : " + str(round(prediction[0][5],3)))
                          print("Neutral : " + str(round(prediction[0][6],3)))
  
  
                          # Rectangle around the face
                          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                      
                          cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                   
                          for (j, k) in shape:
                              cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)'''

                        shape1 = predictor(gray, rect)
                        shape1 = face_utils.shape_to_np(shape1)
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0            
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        cv2.drawContours(img1, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(img1, [rightEyeHull], -1, (0, 255, 0), 1)
                        if ear < EYE_AR_THRESH:
                            COUNTER += 1

                            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                cv2.putText(img1, "DROWSINESS ALERT!", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                                if not ALARM_ON:
                                    ALARM_ON = True                                    
                                    t = Thread(target=sound_alarm)
                                    t.deamon = True
                                    t.start()                            
                        else:
                            COUNTER = 0
                            ALARM_ON = False
                        '''
                        # 1. Add prediction probabilities
                        cv2.putText(frame, "----------------",(40,100 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
                        cv2.putText(frame, "Emotional report : Face #" + str(i+1),(40,120 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
                        cv2.putText(frame, "Angry : " + str(round(prediction[0][0],3)),(40,140 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
                        cv2.putText(frame, "Disgust : " + str(round(prediction[0][1],3)),(40,160 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
                        cv2.putText(frame, "Fear : " + str(round(prediction[0][2],3)),(40,180 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
                        cv2.putText(frame, "Happy : " + str(round(prediction[0][3],3)),(40,200 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
                        cv2.putText(frame, "Sad : " + str(round(prediction[0][4],3)),(40,220 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
                        cv2.putText(frame, "Surprise : " + str(round(prediction[0][5],3)),(40,240 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
                        cv2.putText(frame, "Neutral : " + str(round(prediction[0][6],3)),(40,260 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)'''
                        
                        if prediction_result == 0 :
                            cv2.putText(img1, "Angry",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif prediction_result == 1 :
                            cv2.putText(img1, "Disgust",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif prediction_result == 2 :
                            cv2.putText(img1, "Fear",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif prediction_result == 3 :
                            cv2.putText(img1, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif prediction_result == 4 :
                            cv2.putText(img1, "Sad",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif prediction_result == 5 :
                            cv2.putText(img1, "Surprise",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else :
                            cv2.putText(img1, "Neutral",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        cv2.drawContours(img1, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(img1, [rightEyeHull], -1, (0, 255, 0), 1)
                        nose = shape[nStart:nEnd]
                        noseHull = cv2.convexHull(nose)
                        cv2.drawContours(img1, [noseHull], -1, (0, 255, 0), 1)
                        mouth = shape[mStart:mEnd]
                        mouthHull = cv2.convexHull(mouth)
                        cv2.drawContours(img1, [mouthHull], -1, (0, 255, 0), 1)
                        jaw = shape[jStart:jEnd]
                        jawHull = cv2.convexHull(jaw)
                        cv2.drawContours(img1, [jawHull], -1, (0, 255, 0), 1)                        
                        ebr = shape[ebrStart:ebrEnd]
                        ebrHull = cv2.convexHull(ebr)
                        cv2.drawContours(img1, [ebrHull], -1, (0, 255, 0), 1)
                        ebl = shape[eblStart:eblEnd]
                        eblHull = cv2.convexHull(ebl)
                        cv2.drawContours(img1, [eblHull], -1, (0, 255, 0), 1)

                        '''modelx=tf.keras.models.load_model("64x3-CNN.model")

                        prediction=modelx.predict([prepare('test2.jpg')])
                        print(prediction)'''
                    except:
                        pass
                
                cv2.putText(img1,'Number of Faces : ' + str(len(rects)),(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)
                cv2.imshow('Video', img1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.release()
                    break

    # under work function
    # captures the video and sends the video input to the show_webcam function
    vs=cv2.VideoCapture(0)    
    show_webcam(vs)

def child_male(s,gender,age):
    from PIL import ImageTk, Image
    import webbrowser
    def callback(url):
        webbrowser.open_new(url)
    newWindow = Toplevel(window) 
    
    newWindow.title("boys")
    newWindow.geometry("300x300")
    newWindow.configure(background='grey')
    head=Label(newWindow,text='Top picks for you')
    head.configure(font=('Courier',30))
    head.place(x=800,y=0)

    head2=Label(newWindow,text='Gender = '+gender)
    head2.configure(font=('Courier',20))
    head2.place(x=1350,y=720)
    head3=Label(newWindow,text='Approximate age = '+age)
    head3.configure(font=('Courier',20))
    head3.place(x=1300,y=760)

    ur_pic = s
    ur_pic_img = Image.open(ur_pic)
    ur_pic_img = ur_pic_img.resize((400,300), Image.ANTIALIAS)
    ur_pic_photoImg =  ImageTk.PhotoImage(ur_pic_img)
    ur_pic_panel = Label(newWindow, image = ur_pic_photoImg)
    ur_pic_panel.place(x=1250,y=400)


    path = 'IMG-3630.jpg'
    img = Image.open(path)
    img = img.resize((400,300), Image.ANTIALIAS)
    photoImg =  ImageTk.PhotoImage(img)
    panel = Label(newWindow, image = photoImg)
    panel.place(x=100,y=50)
    link = Label(newWindow, text="look out for more clothes", fg="blue", cursor="hand2")
    link.place(x=550,y=200)
    link.config(font=('Courier',20))
    link.bind("<Button-1>", lambda e: callback("https://www.amazon.in/s?k=clothes+for+kids+boys&i=apparel&crid=34FBULXMLNYCS&sprefix=clothes+for+kids%2Capparel%2C292&ref=nb_sb_ss_ac-a-p_2_16"))
    

    path1 = 'IMG_3644.jpg'
    img1 = Image.open(path1)
    img1 = img1.resize((400,300), Image.ANTIALIAS)
    photoImg1=  ImageTk.PhotoImage(img1)
    panel1 = Label(newWindow, image = photoImg1)
    panel1.place(x=100,y=410)
    link1 = Label(newWindow, text="look out for more toys", fg="blue", cursor="hand2")
    link1.place(x=550,y=550)
    link1.config(font=('Courier',20))
    link1.bind("<Button-1>", lambda e: callback("https://www.amazon.com/s?k=toys+for+kids+boys&ref=nb_sb_noss_1"))
    
    path2 = 'IMG_3645.jpg'
    img2 = Image.open(path2)
    img2 = img2.resize((400,300), Image.ANTIALIAS)
    photoImg2=  ImageTk.PhotoImage(img2)
    panel2 = Label(newWindow, image = photoImg2)
    panel2.place(x=100,y=770)
    link2 = Label(newWindow, text="look out for more watches", fg="blue", cursor="hand2")
    link2.place(x=550,y=920)
    link2.config(font=('Courier',20))
    link2.bind("<Button-1>", lambda e: callback("https://www.amazon.com/s?k=watches+for+kids+boys&ref=nb_sb_noss_1"))
    
    kill_button=Button(newWindow,text="Go back",command=newWindow.destroy)
    kill_button.config(font=('Courier',20))
    kill_button.place(x=1600,y=10)

    newWindow.mainloop()

def child_female(s,gender,age):
    from PIL import ImageTk, Image
    import webbrowser
    def callback(url):
        webbrowser.open_new(url)
    newWindow = Toplevel(window) 
    
    newWindow.title("girls")
    newWindow.geometry("300x300")
    newWindow.configure(background='grey')
    head=Label(newWindow,text='Top picks for you')
    head.configure(font=('Courier',30))
    head.place(x=800,y=0)

    head2=Label(newWindow,text='Gender = '+gender)
    head2.configure(font=('Courier',20))
    head2.place(x=1350,y=720)
    head3=Label(newWindow,text='Approximate age = '+age)
    head3.configure(font=('Courier',20))
    head3.place(x=1300,y=760)

    ur_pic = s
    ur_pic_img = Image.open(ur_pic)
    ur_pic_img = ur_pic_img.resize((400,300), Image.ANTIALIAS)
    ur_pic_photoImg =  ImageTk.PhotoImage(ur_pic_img)
    ur_pic_panel = Label(newWindow, image = ur_pic_photoImg)
    ur_pic_panel.place(x=1250,y=400)


    path = 'IMG_3636.jpg'
    img = Image.open(path)
    img = img.resize((400,300), Image.ANTIALIAS)
    photoImg =  ImageTk.PhotoImage(img)
    panel = Label(newWindow, image = photoImg)
    panel.place(x=100,y=50)
    link = Label(newWindow, text="look out for more clothes", fg="blue", cursor="hand2")
    link.place(x=550,y=200)
    link.config(font=('Courier',20))
    link.bind("<Button-1>", lambda e: callback("https://www.amazon.com/s?k=clothes+for+girls+6-7+years+old&crid=1KH05VWM85166&sprefix=clothes+for+girls%2Caps%2C353&ref=nb_sb_ss_i_3_17"))
    

    path1 = 'Kids-Makeup-Toys-Girls-Games-Baby-Cosmetics-Pretend-Play-Set-Hairdressing-Make-Up-Beauty-Toy-For.jpg'
    img1 = Image.open(path1)
    img1 = img1.resize((400,300), Image.ANTIALIAS)
    photoImg1=  ImageTk.PhotoImage(img1)
    panel1 = Label(newWindow, image = photoImg1)
    panel1.place(x=100,y=410)
    link1 = Label(newWindow, text="look out for more toys", fg="blue", cursor="hand2")
    link1.place(x=550,y=550)
    link1.config(font=('Courier',20))
    link1.bind("<Button-1>", lambda e: callback("https://www.amazon.com/s?k=toys+for+girls+6-7+years+old&ref=nb_sb_noss_2"))
    
    path2 = 'paris-new-stylish-watch-for-girls-500x500.jpg'
    img2 = Image.open(path2)
    img2 = img2.resize((400,300), Image.ANTIALIAS)
    photoImg2=  ImageTk.PhotoImage(img2)
    panel2 = Label(newWindow, image = photoImg2)
    panel2.place(x=100,y=770)
    link2 = Label(newWindow, text="look out for more watches", fg="blue", cursor="hand2")
    link2.place(x=550,y=920)
    link2.config(font=('Courier',20))
    link2.bind("<Button-1>", lambda e: callback("https://www.amazon.com/s?k=watches+for+little+girls&crid=353KIC68L1ACS&sprefix=watches+for+%2Caps%2C351&ref=nb_sb_ss_i_4_12"))
    
    kill_button=Button(newWindow,text="Go back",command=newWindow.destroy)
    kill_button.config(font=('Courier',20))
    kill_button.place(x=1600,y=10)

    newWindow.mainloop()

def ad_male(s,gender,age):
    from PIL import ImageTk, Image
    import webbrowser
    def callback(url):
        webbrowser.open_new(url)
    newWindow = Toplevel(window) 
    
    newWindow.title("Men")
    newWindow.geometry("300x300")
    newWindow.configure(background='grey')
    head=Label(newWindow,text='Top picks for you')
    head.configure(font=('Courier',30))
    head.place(x=800,y=0)
    head2=Label(newWindow,text='Gender = '+gender)
    head2.configure(font=('Courier',20))
    head2.place(x=1350,y=720)
    head3=Label(newWindow,text='Approximate age = '+age)
    head3.configure(font=('Courier',20))
    head3.place(x=1300,y=760)

    ur_pic = s
    ur_pic_img = Image.open(ur_pic)
    ur_pic_img = ur_pic_img.resize((400,300), Image.ANTIALIAS)
    ur_pic_photoImg =  ImageTk.PhotoImage(ur_pic_img)
    ur_pic_panel = Label(newWindow, image = ur_pic_photoImg)
    ur_pic_panel.place(x=1250,y=400)

    path = 'IMG_3641.jpg'
    img = Image.open(path)
    img = img.resize((400,300), Image.ANTIALIAS)
    photoImg =  ImageTk.PhotoImage(img)
    panel = Label(newWindow, image = photoImg)
    panel.place(x=100,y=50)
    link = Label(newWindow, text="look out for more clothes", fg="blue", cursor="hand2")
    link.place(x=550,y=200)
    link.config(font=('Courier',20))
    link.bind("<Button-1>", lambda e: callback("https://www.amazon.com/s?k=clothes+for+men&ref=nb_sb_noss_1"))
    
    path1 = 'watch.png'
    img1 = Image.open(path1)
    img1 = img1.resize((400,300), Image.ANTIALIAS)
    photoImg1=  ImageTk.PhotoImage(img1)
    panel1 = Label(newWindow, image = photoImg1)
    panel1.place(x=100,y=410)
    link1 = Label(newWindow, text="look out for more watches", fg="blue", cursor="hand2")
    link1.place(x=550,y=550)
    link1.config(font=('Courier',20))
    link1.bind("<Button-1>", lambda e: callback("https://www.amazon.com/s?k=watches+for+men&crid=27TRXYHRP4RH3&sprefix=watches+%2Caps%2C346&ref=nb_sb_ss_i_1_8"))
    
    path2 = 'men-s-sport-shoes-500x500.jpg'
    img2 = Image.open(path2)
    img2 = img2.resize((400,300), Image.ANTIALIAS)
    photoImg2=  ImageTk.PhotoImage(img2)
    panel2 = Label(newWindow, image = photoImg2)
    panel2.place(x=100,y=770)
    link2 = Label(newWindow, text="look out for more shoes", fg="blue", cursor="hand2")
    link2.place(x=550,y=920)
    link2.config(font=('Courier',20))
    link2.bind("<Button-1>", lambda e: callback("https://www.amazon.com/s?k=shoes+for+men&ref=nb_sb_noss_1"))
    
    kill_button=Button(newWindow,text="Go back",command=newWindow.destroy)
    kill_button.config(font=('Courier',20))
    kill_button.place(x=1600,y=10)

    newWindow.mainloop()

def ad_female(s,gender,age):
    from PIL import ImageTk, Image
    import webbrowser
    def callback(url):
        webbrowser.open_new(url)
    newWindow = Toplevel(window) 
    
    newWindow.title("women")
    newWindow.geometry("300x300")
    newWindow.configure(background='grey')
    head=Label(newWindow,text='Top picks for you')
    head.configure(font=('Courier',30))
    head.place(x=800,y=0)

    head2=Label(newWindow,text='Gender = '+gender)
    head2.configure(font=('Courier',20))
    head2.place(x=1350,y=720)
    head3=Label(newWindow,text='Approximate age = '+age)
    head3.configure(font=('Courier',20))
    head3.place(x=1300,y=760)

    ur_pic = s
    ur_pic_img = Image.open(ur_pic)
    ur_pic_img = ur_pic_img.resize((400,300), Image.ANTIALIAS)
    ur_pic_photoImg =  ImageTk.PhotoImage(ur_pic_img)
    ur_pic_panel = Label(newWindow, image = ur_pic_photoImg)
    ur_pic_panel.place(x=1250,y=400)


    path = '2.jpg'
    img = Image.open(path)
    img = img.resize((400,300), Image.ANTIALIAS)
    photoImg =  ImageTk.PhotoImage(img)
    panel = Label(newWindow, image = photoImg)
    panel.place(x=100,y=50)
    link = Label(newWindow, text="look out for more clothes", fg="blue", cursor="hand2")
    link.place(x=550,y=200)
    link.config(font=('Courier',20))
    link.bind("<Button-1>", lambda e: callback("https://www.amazon.com/s?k=traditional+dresses+for+women&ref=nb_sb_noss_1"))
    

    path1 = '2017_10$blog_Global Women Cosmetics Market.jpg_17_Oct_2017_071911923.jpg'
    img1 = Image.open(path1)
    img1 = img1.resize((400,300), Image.ANTIALIAS)
    photoImg1=  ImageTk.PhotoImage(img1)
    panel1 = Label(newWindow, image = photoImg1)
    panel1.place(x=100,y=410)
    link1 = Label(newWindow, text="look out for more cosmetics", fg="blue", cursor="hand2")
    link1.place(x=550,y=550)
    link1.config(font=('Courier',20))
    link1.bind("<Button-1>", lambda e: callback("https://www.amazon.com/s?k=cosmeticsfor+women&ref=nb_sb_noss_2"))
    
    path2 = '712LwwAdKSL._AC_UL1500_.jpg'
    img2 = Image.open(path2)
    img2 = img2.resize((400,300), Image.ANTIALIAS)
    photoImg2=  ImageTk.PhotoImage(img2)
    panel2 = Label(newWindow, image = photoImg2)
    panel2.place(x=100,y=770)
    link2 = Label(newWindow, text="look out for more heels", fg="blue", cursor="hand2")
    link2.place(x=550,y=920)
    link2.config(font=('Courier',20))
    link2.bind("<Button-1>", lambda e: callback("https://www.amazon.com/s?k=heelsfor+women&ref=nb_sb_noss_2"))
    
    kill_button=Button(newWindow,text="Go back",command=newWindow.destroy)
    kill_button.config(font=('Courier',20))
    kill_button.place(x=1600,y=10)

    newWindow.mainloop()

def suggest(s):
    import math
    import argparse
    def highlightFace(net, frame, conf_threshold=0.7):
        frameOpencvDnn=frame.copy()
        frameHeight=frameOpencvDnn.shape[0]
        frameWidth=frameOpencvDnn.shape[1]
        blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections=net.forward()
        faceBoxes=[]
        for i in range(detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>conf_threshold:
                x1=int(detections[0,0,i,3]*frameWidth)
                y1=int(detections[0,0,i,4]*frameHeight)
                x2=int(detections[0,0,i,5]*frameWidth)
                y2=int(detections[0,0,i,6]*frameHeight)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn,faceBoxes


    parser=argparse.ArgumentParser()
    parser.add_argument('--image')

    args=parser.parse_args()

    faceProto="opencv_face_detector.pbtxt"
    faceModel="opencv_face_detector_uint8.pb"
    ageProto="age_deploy.prototxt"
    ageModel="age_net.caffemodel"
    genderProto="gender_deploy.prototxt"
    genderModel="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(20-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    frame=cv2.imread(s)
    padding=20

        
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        return f'{gender}',f'{age[1:-1]}'
    
def amazon():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    cv2.imwrite('gen.jpg',frame)
    import test
    gender,age=suggest('gen.jpg')
    al=age.split('-')
    ax=int(al[1])
    if gender=='Male':
        if ax<=10:
            child_male('gen.jpg',gender,age)
        else:
            ad_male('gen.jpg',gender,age)
    elif gender=='Female':    
        if ax<=10:
            child_female('gen.jpg',gender,age)
        else:
            ad_female('gen.jpg',gender,age)

def recognition():
            
    label=Label(window,text='Access granted')
    a=1
    label.config(font=('Courier',64))
    label.place(x=600,y=500)
    label.after(10000 , lambda: label.destroy())

    btn1=Button(window, text = 'Activate system', bd = '5', 
                command = work) 
    btn1.config(font=('Courier',35),fg='red') 
    btn1.place(x=200,y=300)
    
    btn2=Button(window, text='Need any shopping suggestions?',bd=5,
                command=amazon)

    btn2.config(font=('Courier',35),fg='red') 
    btn2.place(x=1000,y=300)
    

    btn3=Button(window, text='Exit',bd=5,
                command=window.destroy)

    btn3.config(font=('Courier',25),fg='blue') 
    btn3.place(x=1650,y=900)
    
lbl1=Label(window,text='WELCOME')
lbl1.config(bg='grey',fg='black',font=('',70))
lbl1.place(x=700,y=0)
btn = Button(window, text = 'Click here to start engine', bd = '5', 
                          command = recognition)  
btn.config(font=('Courier',35),fg='red') 
btn.place(x=550,y=150)     
window.mainloop()


