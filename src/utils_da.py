# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import augmentation.live_augmentation as live_aug
import config as conf

import utils.help_functions as hlpf


def gray2rgb_set(imgset):
    imgset2 = np.zeros((imgset.shape[0], imgset.shape[1], imgset.shape[2], 3),
                       dtype=np.float32)
    for ch in range(3):
        imgset2[:,:,:,ch] = imgset[:,:,:,0]
    return imgset2

# def ms2rgb_set(imgset):
#     imgset2 = np.zeros((imgset.shape[0], imgset.shape[1], imgset.shape[2], 3),
#                        dtype=np.float32)
#     for ch in range(3):
#         imgset2[:,:,:,ch] = imgset[:,:,:,ch]
#     return imgset2


def ms2rgb_set(ms_batch):
    
    new_batch = hlpf.batch_ms2rgb(ms_batch)
       
    return new_batch


def rgb2gray_set(imgset):
    from utils import image_utils
    imgset2 = image_utils.image_colour_convert(imgset, 'greyscale')
    return imgset2


def load_trained_model(sess, checkpoint_path):
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    
    
def load_scaled_image(image_path, color_mode='rgb'):

    img = live_aug.load_img(image_path, color_mode=color_mode)
    img_scaled = live_aug.processImg(img, color_mode)
    return img_scaled


def load_unscaled_image(image, color_mode='rgb'):

    #return live_aug.load_img(image_path, color_mode=color_mode)
    return live_aug.unprocessImg(image, color_mode)




def load_scaled_image_gray(image_path):
    img = live_aug.load_img(image_path, grayscale=True)
    img_scaled = live_aug.divide_by_max(img, 255.0)
    return img_scaled


def get_scaled_image(image_path, color_mode):
    if color_mode=='rgb' or color_mode=="multispectral":
        img_input = load_scaled_image(image_path, color_mode)
        
    else:
        img_input = load_scaled_image_gray(image_path)
    return img_input

