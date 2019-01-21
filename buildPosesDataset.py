import cv2
import os
from os.path import isfile, join
import numpy as np
from random import randint
from sklearn.utils import shuffle

def read_data(req_poses):
    count_im = 0
    count_classes = 0
    poses = os.listdir('Poses/')
    if(req_poses[0]=='all'):
        req_poses = poses.copy()
    for pose in poses:      
        if pose in req_poses:    
            print(">> Working on pose : " + pose)
            subdirs = os.listdir('Poses/' + pose + '/')
            count_classes += 1
            for subdir in subdirs:
                files = os.listdir('Poses/' + pose + '/' + subdir + '/')
                print(">> Working on examples : " + subdir)
                for file in files:
                    if(file.endswith(".png")):
                        count_im += 1
    print(str(count_classes)  + ' classes')
    print(str(count_im) + ' images')
    x = np.empty(shape=(count_im, 28, 28, 1))
    y = np.empty(count_im)

    count_im = 0
    count_classes = 0
    poses = os.listdir('Poses/')
    for pose in poses:
        if pose in req_poses:
            print(">> Working on pose : " + pose)
            subdirs = os.listdir('Poses/' + pose + '/') 
            for subdir in subdirs:
                files = os.listdir('Poses/' + pose + '/' + subdir + '/')
                print(">> Working on examples : " + subdir)
                for file in files:
                    if(file.endswith(".png")):
                        path = 'Poses/' + pose + '/' + subdir + '/' + file
                        # Read image
                        im = cv2.imread(path)
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        im = im.astype(dtype="float64")
                        im = np.reshape(im, (28, 28, 1))
                        x[count_im][:][:][:] = im
                        y[count_im] = count_classes
                        count_im += 1
            count_classes += 1
    x = x/255

    return x, y

def load_data(poses=['all']):
   x,y = read_data(poses)
   x,y = shuffle(x, y, random_state=0)
   x_train, y_train, x_test, y_test = split_data(x,y)
   return x_train, y_train, x_test, y_test

def split_data(x,y,split=0.85):
    maxIndex = int(split*x.shape[0])
    x_train = x[:maxIndex][:][:][:]
    x_test = x[maxIndex:][:][:][:]
    y_train = y[:maxIndex]
    y_test = y[maxIndex:]
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    print(y_train.shape, y_test.shape)