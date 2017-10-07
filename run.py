#!/usr/bin/env python 
import sys
import os
import time
import subprocess as sp
import itertools
## CV
import cv2
import skvideo.io
## Model
import numpy as np
import tensorflow as tf
## Tools
import utils
## Parameters
import params ## you can modify the content of params.py


## Test epoch
epoch_ids = [10]
## Load model
model = utils.get_model()

## Preprocess
def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    '''
    ## Chop off 1/3 from the top and cut bottom 150px(which contains the head of car)
    shape = img.shape
    img = img[int(shape[0]/3):shape[0]-150, 0:shape[1]]
    ## Resize the image
    img = cv2.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h), interpolation=cv2.INTER_AREA)
    ## Return the image sized as a 4D array
    return np.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h, params.FLAGS.img_c))
    '''
    ratio = params.FLAGS.img_h / params.FLAGS.img_w
    h1, h2 = int(img.shape[0] / 2), img.shape[0] - 150
    w = (h2 - h1) / ratio
    padding = int(round((img.shape[1] - w) / 2))
    img = img[h1:h2, padding:-padding]
    # Resize the image
    img = cv2.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h),
                     interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img


## Process video
for epoch_id in epoch_ids:
    print('---------- processing video for epoch {} ----------'.format(epoch_id))
    vid_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
    assert os.path.isfile(vid_path)
    frame_count = utils.frame_count(vid_path)
    #cap = cv2.VideoCapture(vid_path)
    cap = skvideo.io.vreader(vid_path)

    machine_steering = []

    print('performing inference...')
    time_start = time.time()
    #for frame_id in range(frame_count):
    for img in cap:
        #ret, img = cap.read()
        #assert img
        ## you can modify here based on your model
        img = img_pre_process(img)
        img = img[None,:,:,:]
        deg = float(model.predict(img, batch_size=1))
        deg = float(int(deg*2+0.5))/2.0
        machine_steering.append(deg)

    del cap

    fps = frame_count / (time.time() - time_start)
    
    print('completed inference, total frames: {}, average fps: {} Hz'.format(frame_count, round(fps, 1)))
    
    print('performing visualization...')
    utils.visualize(epoch_id, machine_steering, params.out_dir,
                        verbose=True, frame_count_limit=None)
    
    
