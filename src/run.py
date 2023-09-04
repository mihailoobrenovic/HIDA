# --usecase u_hida --source resisc --target eurosat --exp_name u_hida_1
# --usecase ss_hida --source resisc --target eurosat --threshold 1_25 --exp_name ss_hida_1_25_1 
# --usecase ss_hida --source eurosat --target resisc --threshold 1_25 --exp_name ss_hida_1_25_2 

# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import os

from network_parameters import params
import training_functions as train
import config as conf

import parse as p
##prevent Tensorflow from taking up entire GPU memory
cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
session = tf.Session(config=cfg)

start = time.time()

num_runs = p.num_runs


color_mode_s = conf.color_mode_s
color_mode_t = conf.color_mode_t

color_mode = color_mode_s

for i in range(num_runs):
    if num_runs > 1 or p.idx == -1:
        conf.repetition_no = i+1
    else:
        conf.repetition_no = p.idx+1
    train.train_adapter(
        params, validate_all=True, save_on=50, checkpoint_on=p.checkpoint_on, 
        test_wd=True, color_mode_s=color_mode_s, color_mode_t=color_mode_t, 
        two_fes=True, include_silhouette=True, tb=True)
        
end = time.time()
print("Running time: ", end-start)
