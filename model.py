
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import cv2
import img_gen
import numpy as np
import time

class_names =  ["empty","cup","glue","keyboard","marker","mouse","pen"]
nClasses = len(class_names)
#referenced https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=img_gen.input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def classify(video):
    cap = cv2.VideoCapture(video)
    model = createModel()
    model.load_weights("model1.h5")
    while cap.isOpened():
        ret,frame = cap.read()
        if(not ret):
            break
        start_time = time.time()
        img = img_gen.convertimgfornet(frame)
        x  = np.asarray([img,])
        out = model.predict(x)
        max = 0
        cls_idx = 0
        for i,d in enumerate(out[0]):
            if(max<d):
                max = d
                cls_idx = i
        fps = int(1/(time.time()-start_time))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'cls:{} prob:{}% fps:{}'.format(class_names[cls_idx],int(max*100),fps),(10,50), font, 0.6,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow("result",frame)
        k = cv2.waitKey(10)
        if(k==ord("s")):
            cv2.imwrite("result.jpg",frame)




classify("./dataset/KakaoTalk_Video_20181030_1923_33_745.mp4")
        

