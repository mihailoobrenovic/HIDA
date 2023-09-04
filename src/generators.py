# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle as pkl
import sys

import config as conf
import parse as p
from augmentation.live_augmentation import ImageDataGenerator

np.random.seed(None)


def get_input_path(source):
    if source:
        inputpath = conf.inputpath_s
    else:
        inputpath = conf.inputpath_t
    return inputpath


# Create batch generators for space data

def get_space_train_generator(inp_shape, otp_shape, number_of_classes, 
                                batch_size, color_mode='rgb', source=True, 
                                classes=None, wdgrl_train_flow=False, train_test=False):
    train_gen = ImageDataGenerator(
            standardise_sample=True, samplewise_normalise= True, 
            nb_classes=number_of_classes, categoricaltarget=False, **conf.train_augmentation_args)
    
    repetition_no = conf.repetition_no
    # print("Repetition", repetition_no)
    
    train_dir_name = 'train'
    if wdgrl_train_flow or (not source and train_test):
        train_dir_name += "_wdgrl" 
    elif p.train_dir:
        train_dir_name += "_" +  str(p.train_dir)
    else:
        train_dir_name += "_" +  str(repetition_no)
    
    inputpath = get_input_path(source)
    if source:
        domain = conf.staining_s# + "_" + color_mode
    else:
        domain = conf.staining_t# + "_" + color_mode
    if domain in conf.means.keys():
        # print('Already has mean and stddev')
        mean, stddev = conf.means[domain], conf.stddevs[domain]
        train_gen.dataset_mean, train_gen.dataset_std = mean, stddev
        train_flow = train_gen.flow_from_directory(
                os.path.join(inputpath, train_dir_name), 
                img_target_size=(inp_shape[0], inp_shape[1]),
                gt_target_size=(otp_shape[0], otp_shape[1]), 
                color_mode=color_mode, batch_size=batch_size, shuffle=True,
                dataset_mean=mean, dataset_std=stddev, classes=classes, augmentationclassblock=conf.train_augmentationclassblock)
    else:
        print('No mean and stddev')
        train_flow = train_gen.fit_and_flow_from_directory(
                os.path.join(inputpath, train_dir_name), 
                img_target_size=(inp_shape[0], inp_shape[1]), 
                gt_target_size=(otp_shape[0], otp_shape[1]),
                color_mode=color_mode, batch_size=batch_size, shuffle=True, classes=classes, augmentationclassblock=conf.train_augmentationclassblock)
        mean, stddev = train_gen.get_fit_stats()
        print('Mean: ', mean, 'stddev: ', stddev)
        
    print(domain, ":  train class_indices: ", train_flow.class_indices)
    return train_flow, mean, stddev


def get_space_val_generator(inp_shape, otp_shape, number_of_classes,  
                              batch_size, mean, stddev, color_mode='rgb',
                              source=True, classes=None, folder='test'):
    inputpath = get_input_path(source)
    validation_gen = ImageDataGenerator(
            standardise_sample= True, samplewise_normalise= True,
            nb_classes=number_of_classes, categoricaltarget=False, **conf.valid_augmentation_args)
    
    valid_flow = validation_gen.flow_from_directory(
            os.path.join(inputpath, folder),
            img_target_size=(inp_shape[0], inp_shape[1]),
            gt_target_size=(otp_shape[0], otp_shape[1]),
            color_mode=color_mode,batch_size=batch_size, shuffle=True,
            dataset_mean=mean, dataset_std=stddev, classes=classes, augmentationclassblock=conf.valid_augmentationclassblock)
    print(color_mode, ":  test class_indices: ", valid_flow.class_indices)
    return valid_flow



def get_train_generator(inp_shape, otp_shape, num_class, batch_size, 
                        color_mode, source=True, 
                        wdgrl_train_flow=False, train_test=False):
    
    if source:#********************
        classes = conf.classes_s
    else:
        classes = conf.classes_t
    
    
    mean = stddev = None

    if conf.dataset == 'space':
        train_flow, mean, stddev = get_space_train_generator(
                inp_shape, otp_shape, num_class,  batch_size // 2, 
                color_mode=color_mode, source=source, classes=classes, 
                wdgrl_train_flow=wdgrl_train_flow, train_test=train_test)

    else:
        sys.exit()
    return train_flow, mean, stddev



def get_train_generators(inp_shape_s, inp_shape_t, otp_shape, num_class, 
                         batch_size, color_mode_s, color_mode_t,
                         wdgrl_train_flow=False, train_test=False):
    
    print("\n****source_train_flow")
    train_flow_s, mean_s, stddev_s = get_train_generator(
            inp_shape_s, otp_shape, num_class, batch_size, color_mode_s,
            source=True)
    print("\n")
    
    
    print("\n****target_train_flow")
    
    train_flow_t, mean_t, stddev_t = get_train_generator(
            inp_shape_t, otp_shape, num_class, batch_size, color_mode_t,
            source=False, train_test=train_test)
    print("\n")
   
    

    if not wdgrl_train_flow:
        return train_flow_s, train_flow_t, mean_s, mean_t, stddev_s, stddev_t
    
    elif wdgrl_train_flow:
        print("\n****target wdgrl_train_flow")
        wdgrl_train_flow_t, wdgrl_mean_t, wdgrl_stddev_t = get_train_generator(
            inp_shape_t, otp_shape, num_class, batch_size, color_mode_t,
            source=False, wdgrl_train_flow=wdgrl_train_flow) 
        print("\n")
        return train_flow_s, train_flow_t, wdgrl_train_flow_t, mean_s, mean_t, wdgrl_mean_t, stddev_s, stddev_t,  wdgrl_stddev_t 
    


def get_val_generator(inp_shape, otp_shape, num_class, batch_size, color_mode,
                      source, mean=None, stddev=None, folder='test'):
    
    
    if source:#********************
        classes = conf.classes_s
    else:
        classes = conf.classes_t
    
    
    if conf.dataset == 'space':
         valid_flow = get_space_val_generator(
                inp_shape, otp_shape, num_class, batch_size // 2, 
                mean, stddev, color_mode=color_mode, source=source, classes=classes, folder=folder)
    else:
        sys.exit()
    return valid_flow
        
        
def get_val_generators(inp_shape_s, inp_shape_t, otp_shape, num_class, 
                       batch_size, color_mode_s, color_mode_t, mean_s=None,
                       mean_t=None, stddev_s=None, stddev_t=None,
                       folder='test'):
    valid_flow_s = get_val_generator(
            inp_shape_s, otp_shape, num_class, batch_size, color_mode_s,
            source=True, mean=mean_s, stddev=stddev_s, 
            folder=folder)
    valid_flow_t = get_val_generator(
            inp_shape_t, otp_shape, num_class, batch_size, color_mode_t,
            source=False, mean=mean_t, stddev=stddev_t, 
            folder=folder)
    return valid_flow_s, valid_flow_t


def get_1_domain_train_and_val_generators(
        inp_shape, otp_shape, num_class, batch_size, color_mode, source):
    train_flow, mean, stddev = get_train_generator(
            inp_shape, otp_shape, num_class, batch_size, color_mode,
            source=source)
    valid_flow = get_val_generator(
            inp_shape, otp_shape, num_class, batch_size, color_mode,
            source=source, mean=mean, stddev=stddev)
    return train_flow, valid_flow


def get_all_generators(inp_shape_s, inp_shape_t, otp_shape, num_class, 
                       batch_size, color_mode_s, color_mode_t, train_test=False):
    mean_s = mean_t = 0
    stddev_s = stddev_t = 1
    # train_flow_s, train_flow_t, mean_s, mean_t, stddev_s, stddev_t = \
    #     get_train_generators(inp_shape_s, inp_shape_t, otp_shape, num_class, 
    #                          batch_size, color_mode_s, color_mode_t,
    #                          train_test=train_test)
    valid_flow_s, valid_flow_t = get_val_generators(
            inp_shape_s, inp_shape_t, otp_shape, num_class, batch_size, 
            color_mode_s, color_mode_t, mean_s=mean_s, mean_t=mean_t, 
            stddev_s=stddev_s, stddev_t=stddev_t)
    
    # return train_flow_s, train_flow_t, valid_flow_s, valid_flow_t
    return valid_flow_s, valid_flow_t



    
