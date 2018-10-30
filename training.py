#training code

import cv2
import img_gen


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import model






import numpy as np



def train():


    train,test =  img_gen.loadDataset()
    train_data = []
    train_labels_one_hot = []
    test_data = []
    test_labels_one_hot = []
    for dat in train:
        train_data.append(dat[0])
        train_labels_one_hot.append(dat[1])
    for dat in test:
        test_data.append(dat[0])
        test_labels_one_hot.append(dat[1])

    train_data = np.asarray(train_data )
    train_labels_one_hot = np.asarray(train_labels_one_hot )
    test_data = np.asarray(test_data )
    test_labels_one_hot = np.asarray(test_labels_one_hot )

    model1 = model.createModel()

    batch_size = 15
    epochs = 50
    
    LoadPretrained = False
    if(LoadPretrained):
        model1.load_weights("model1.h5")

    history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, 
                    validation_data=(test_data, test_labels_one_hot))

    model1.evaluate(test_data, test_labels_one_hot)
    model1.save_weights("model1.h5")

        

    import matplotlib.pyplot as plt
    # Loss Curves

    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    
    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.waitforbuttonpress()
    input("press enter to finish:")


if (__name__=="__main__"):
    train()