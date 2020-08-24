import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


# -*- coding:utf-8 -*-
from models import model_one_class, dense201_hirar, dense201_hirar_6same20, dense201_hirar_new
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
import numpy as np
import keras
import random
#import cv2
import os
import random
#import matplotlib
#matplotlib.use('AGG')
#import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint,CSVLogger
import tensorflow as tf
import keras.backend as K
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#from schedules import onetenth_4_8_12
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")



def preprocess(inputs):
    '''
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    '''
    inputs /=255.
    #inputs -= 0.5
    #inputs *=2.
    return inputs


def process_batch(lines,img_path,train=True):
    num = len(lines)
    batch = np.zeros((num,224,224,3),dtype='float32')
    x1_labels = np.zeros(num,dtype='int')
    x2_labels = np.zeros(num,dtype='int')
    x3_labels = np.zeros(num,dtype='int')

    for i in range(num):
        path = lines[i].split(',')[0]
        x1_label = lines[i].split(',')[1]
        x2_label = lines[i].split(',')[2]
        x3_label = lines[i].split(',')[-1]
        x3_label = x3_label.strip('\n')
        
        x1_label = int(x1_label)
        x2_label = int(x2_label)
        x3_label = int(x3_label)

        imgs = img_path+path
        #print(imgs)
        #imgs.sort(key=str.lower)
        if train:
            image = Image.open(imgs).convert("RGB")
            #print(image.shape)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.resize((224,224), Image.ANTIALIAS)
            batch[i][:][:][:] = image
            x1_labels[i] = x1_label
            x2_labels[i] = x2_label
            x3_labels[i] = x3_label
        else:
            image = Image.open(imgs).convert("RGB")
            #print(image.shape)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.resize((224,224), Image.ANTIALIAS)
            batch[i][:][:][:] = image
            x1_labels[i] = x1_label
            x2_labels[i] = x2_label
            x3_labels[i] = x3_label
    
    return batch, x1_labels, x2_labels, x3_labels




def generator_train_batch(train_txt,batch_size,num_classes,img_path):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    class_6 = num_classes[0]
    class_20 = num_classes[1]
    class_82 = num_classes[2]

    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num/batch_size)):
            a = i*batch_size
            b = (i+1)*batch_size
            x_train, x1_labels, x2_labels, x3_labels = process_batch(new_line[a:b],img_path,train=True)
            x = preprocess(x_train)
            #x = x_train
            y1 = np_utils.to_categorical(np.array(x1_labels), class_6)
            y2 = np_utils.to_categorical(np.array(x2_labels), class_20)
            y3 = np_utils.to_categorical(np.array(x3_labels), class_82)
            #y = np_utils.to_categorical(np.array(x_labels), num_classes)
            #x = np.transpose(x, (0,3,1,2))
            y = [y1,y2,y3]
            yield x, y


def generator_val_batch(val_txt,batch_size,num_classes,img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    class_6 = num_classes[0]
    class_20 = num_classes[1]
    class_82 = num_classes[2]

    while True:
        new_line = []
        index = [n for n in range(num)]
        #random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test,y1_labels, y2_labels, y3_labels = process_batch(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            #y = np_utils.to_categorical(np.array(y_labels), num_classes)
            y1 = np_utils.to_categorical(np.array(y1_labels), class_6)
            y2 = np_utils.to_categorical(np.array(y2_labels), class_20)
            y3 = np_utils.to_categorical(np.array(y3_labels), class_82)
            test_data = x
            y = [y1,y2,y3]
            yield test_data, y


def main():

    path = '/home/user/3TB_HDD/Database/Yoga_July_09_Final/'#'home/your_path/'
    img_path = path+'yoga_dataset_images_final/'
    train_file = path+'yoga_train.txt'
    test_file = path+'yoga_test.txt'
    #test_test ='yoga_test.txt'
    f1 = open(train_file, 'r')
    f2 = open(test_file, 'r')
    lines = f1.readlines()

    f1.close()
    train_samples = len(lines)
    #print(train_samples)

    lines = f2.readlines()
    f2.close()
    test_samples = len(lines)

    num_classes = [6,20,82]
    batch_size = 32
    epochs = 50

    model = dense201_hirar_new()
    #model.load_weights('weights_betweenhirarModify_lw111_6and20ConnectSame_nopre_mix_.0003.hdf5')


    lr = 0.003 # orig= 0.003
    sgd = SGD(lr=lr, momentum=0.9, nesterov=False)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'], loss_weights=[1,1,1], optimizer= sgd, metrics=['accuracy'])#, 'top_k_categorical_accuracy'])
    #model.compile(loss=['categorical_crossentropy'], optimizer= sgd, metrics=['accuracy'])

    model.summary()

    #'''
    checkpointer = ModelCheckpoint(filepath=path+'weights_betweenhirarModify_lw111_dense32_nopre_mix_.0003.hdf5', verbose=1, save_best_only= True, monitor='val_loss')
    csv_logger= CSVLogger(path+'log_betweenhirarModify_lw111_dense32_nopre_mix_.0003.csv')
    
    model.fit_generator(generator_train_batch(train_file, batch_size, num_classes,img_path),
                          steps_per_epoch=train_samples // batch_size,
                          epochs=epochs,
                          callbacks=[checkpointer, csv_logger],
                          validation_data=generator_val_batch(test_file, batch_size,num_classes,img_path),
                          validation_steps=test_samples // batch_size,
                          verbose=1)
    #'''
    ## Test evaluation
    #model.load_weights(path+'weights_nopre_imagenet_betweenhirarModify_lw111_dense32_Retrain.00003.hdf5')


    score = model.evaluate_generator(generator_val_batch(test_file,8,num_classes,img_path),steps=test_samples // 8, verbose=1)
    print(score)

if __name__ == '__main__':
    main()
