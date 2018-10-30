from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import csv
import random
import os
import numpy
import cv2
#1. video to image conversion code
#2. img to array


def readVideo():
    cap = cv2.VideoCapture("./dataset/KakaoTalk_Video_20181030_1923_33_745.mp4")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    while cap.isOpened():
        i+=1
        ret,frame = cap.read()
        if(not ret):
            break
        if(i%15!=0):
            continue
        print("{}%".format(i*100/length))
        cv2.imwrite("./dataset/{:04d}.jpg".format(i),frame)

def writeImgNames():
    f = open("out.txt","w")
    for i in range(70):
        f.write("{0:04d}.jpg\n".format(i*15))
    f.close()

input_shape = (250,250,3)

def convertimgfornet(img):
    img =cv2.resize(img,(250,250))
    img = numpy.asarray(img,dtype=numpy.float64)
    img /= 255.
    img = img_to_array(img)
    return img

def loadDataset():

    classes = ["empty","cup","glue","keyboard","marker","mouse","pen"]
    data = []
  

    for folder in os.listdir("./dataset/"):
        if(not os.path.isdir("./dataset/"+folder)):
            continue
        label = numpy.zeros(len(classes))
        for i,val in enumerate(classes):
            if(val==folder):
                label[i] = 1
        for file in os.listdir("./dataset/"+folder):
            img = cv2.imread("./dataset/"+folder+"./"+file)
            img = convertimgfornet(img)
            data.append((img,label))


    random.shuffle(data)

    trainset = data[:45]
    testset = data[45:]
    return (trainset,testset)


#loadDataset()
        