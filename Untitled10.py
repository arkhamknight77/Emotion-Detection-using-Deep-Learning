#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch

from torchvision import transforms

import os
import cv2
import numpy as np
from collections import defaultdict

import scikitplot
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential, model_from_json

from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from IPython.display import Image

from keras.preprocessing import image


# In[19]:


label_emotion_mapper = {0:"surprise", 1:"happy", 2:"anger", 3:"sadness", 4:"fear"}


# In[26]:


class_labels=['surprise','happy','anger','sadness','fear']


# In[39]:


# Saving Model weights and json
model_json_file = 'C:/Users/hp/Downloads/model-3.json'
model_weights_file = 'C:/Users/hp/Downloads/model-3.h5'
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_file)


# In[40]:


# default harcascade 
face_cascade = cv2.CascadeClassifier('C:/Users/hp/Downloads/haarcascade_frontalface_default.xml')


# In[41]:



class_labels=['surprise','happy','anger','sadness','fear']
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            image = np.array(roi)  # image is a NumPy array
            roi = np.asarray(image)
            input_data=np.expand_dims(roi,axis=0)
            
            input_data = np.expand_dims(input_data, axis=-1)
            input_data = np.expand_dims(input_data, axis=1)
            input_data = np.tile(input_data, (1, 3, 1, 1, 1))

            preds=loaded_model.predict(input_data)[0]
            label=class_labels[preds.argmax()]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[37]:


path='C:/pradyumn/SAP@NUS/project/test/anger/chrome_0NVe1Cb1BN.png'
img1=cv2.imread('C:/pradyumn/SAP@NUS/project/test/anger/chrome_0NVe1Cb1BN.png')
plt.imshow(img1)
plt.show()
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    print("hi")
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    if np.sum([roi_gray])!=0:
        roi=roi_gray.astype('float')/255.0
        image = np.array(roi)  # image is a NumPy array
        roi = np.asarray(image)
        input_data=np.expand_dims(roi,axis=0)
    
        input_data = np.expand_dims(input_data, axis=-1)
        input_data = np.expand_dims(input_data, axis=1)
        input_data = np.tile(input_data, (1, 3, 1, 1, 1))

        preds=loaded_model.predict(input_data)[0]
        label=class_labels[preds.argmax()]
        print(label)
            


# In[28]:


path='C:/pradyumn/SAP@NUS/project/test/anger/chrome_gJWbqFAKK6.png'
img1=cv2.imread('C:/pradyumn/SAP@NUS/project/test/happy/chrome_gJWbqFAKK6.png')
plt.imshow(img1)
plt.show()
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    print("hi")
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    if np.sum([roi_gray])!=0:
        print("hi")
        roi=roi_gray.astype('float')/255.0
        image = np.array(roi)  # image is a NumPy array
        roi = np.asarray(image)
        input_data=np.expand_dims(roi,axis=0)
    
        input_data = np.expand_dims(input_data, axis=-1)
        input_data = np.expand_dims(input_data, axis=1)
        input_data = np.tile(input_data, (1, 3, 1, 1, 1))

        preds=loaded_model.predict(input_data)[0]
        label=class_labels[preds.argmax()]
        print(label)


# In[29]:


img1=cv2.imread('C:/Users/hp/Downloads/archive (6)/Football players faces dataset/Fifa_player_dataset/Courtois/chrome_GdHkBAeRpS.png')
plt.imshow(img1)
plt.show()
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    if np.sum([roi_gray])!=0:
        roi=roi_gray.astype('float')/255.0
        image = np.array(roi)  # image is a NumPy array
        roi = np.asarray(image)
        input_data=np.expand_dims(roi,axis=0)
    
        input_data = np.expand_dims(input_data, axis=-1)
        input_data = np.expand_dims(input_data, axis=1)
        input_data = np.tile(input_data, (1, 3, 1, 1, 1))

        preds=loaded_model.predict(input_data)[0]
        label=class_labels[preds.argmax()]
        print(label)


# In[ ]:





# In[15]:


cap = cv2.VideoCapture(0)
import copy


while True:
    
    ret, frame = cap.read()
    img = copy.deepcopy(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #img = copy.deepcopy(frame)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB format
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        fc = gray[y:y+h, x:x+w]
        
       
        roi = cv2.resize(fc, (48,48))
        new_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        
        new_roi = new_roi.reshape(1, 3, 48, 48, 1)
        
        #pred = loaded_model.predict(new_roi[np.newaxis, :, :, np.newaxis])
        
        #roi = cv2.resize(fc, (48,48))
        #roi = np.expand_dims(roi, axis=0)  # Add the missing "3" dimension
        #pred = loaded_model.predict(roi)
        
        #roi = cv2.resize(fc, (48,48))
        #roi = roi.reshape(1, 3, 48, 48, 1)  # Reshape the input to the correct shape
        pred = loaded_model.predict(new_roi)
        
        #roi = cv2.resize(fc, (48,48))
        #roi = roi.reshape(1, 1, 48, 48, 1)  # Reshape the input to the correct shape
        #pred = loaded_model.predict(roi)
        '''text_idx=np.argmax(pred)
        text_list = ['happy', 'surprise', 'anger', 'sadness', 'fear']
        if text_idx == 0:
            text= text_list[0]
        if text_idx == 1:
            text= text_list[1]
        elif text_idx == 2:
            text= text_list[2]
        elif text_idx == 3:
            text= text_list[3]
        elif text_idx == 4:
            text= text_list[4]
        '''
        #print(np.argmax(pred))
        text = label_emotion_mapper[np.argmax(pred)]
        cv2.putText(img, text, (x, y-5),
           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
        img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            
    
    cv2.imshow("frame", img)
    key = cv2.waitKey(1) & 0xFF
    if key== ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


# In[14]:


import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
  print('hi')
else:
  print('hello')


# In[ ]:




