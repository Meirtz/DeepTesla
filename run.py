#!/usr/bin/env python 
import sys
import os
import time
import subprocess as sp
import itertools
import pandas as pd
## CV
import cv2
import skvideo.io
import matplotlib.pyplot as plt
#%matplotlib inline
## Model
import numpy as np
import tensorflow as tf
## Tools
import utils
## Parameters
import params ## you can modify the content of params.py

## Test epoch
epoch_ids = range(1,11)
## Load model
model = utils.get_model()

## Preprocess
def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    ## Chop off 1/3 from the top and cut bottom 150px(which contains the head of car)
    shape = img.shape
    img = img[int(shape[0]/3):shape[0]-150, 0:shape[1]]
    ## Resize the image
    img = cv2.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h), interpolation=cv2.INTER_AREA)
    ## Return the image sized as a 4D array
    return np.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h, params.FLAGS.img_c))


## Process video
def extract_frames():
    for epoch_id in epoch_ids:
        print('---------- processing video for epoch {} ----------'.format(epoch_id))
        vid_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
        csv_path = "./epochs/epoch{:0>2d}_steering.csv".format(epoch_id)
        dir_name = "./epochs/epoch{:0>2d}_front_frames".format(epoch_id)
        os.mkdir(dir_name)
        frame_ids = pd.read_csv(csv_path)['ts_micro'].tolist()
        assert os.path.isfile(vid_path)
        frame_count = utils.frame_count(vid_path)
        cap = skvideo.io.vreader(vid_path) #cv2.VideoCapture(vid_path)

        machine_steering = []

        print('performing inference...')
        time_start = time.time()
        frame_count = 0
        for img in cap:
            #assert img
            ## you can modify here based on your model
            #print(img.shape)
            #cv2.imshow('frame',img)
            
            img = img_pre_process(img)
            #img = img[None,:,:,:]
            cv2.imwrite("{}/{}.jpg".format(dir_name, frame_ids[frame_count-1]), img)
            #deg = float(model.predict(img, batch_size=1))
            #machine_steering.append(deg)
            frame_count += 1

        del cap

        fps = frame_count / (time.time() - time_start)
    
        print('completed inference, total frames: {}, average fps: {} Hz'.format(frame_count, round(fps, 1)))
    
        #print('performing visualization...')
        #utils.visualize(epoch_id, machine_steering, params.out_dir,
                            #verbose=True, frame_count_limit=None)
        
def load_dataset():
    image_set = []
    steerings = []
    for epoch_id in epoch_ids:
        csv_path = "./epochs/epoch{:0>2d}_steering.csv".format(epoch_id)
        dir_name = "./epochs/epoch{:0>2d}_front_frames".format(epoch_id)
        filenames = pd.read_csv(csv_path)['ts_micro'].tolist()
        steering = pd.read_csv(csv_path)['wheel'].tolist()
        steerings.extend(steering)
        for img in filenames:
            image_set.append(cv2.imread(dir_name+str(img)+'.jpg'))
    return np.array(image_set), np.array(steerings)       
            
            
#extract_frames()
X, y = load_dataset()
print(X.shape, y.shape)
    
    