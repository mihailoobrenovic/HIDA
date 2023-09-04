# -*- coding: utf-8 -*-
import os
import platform
hostname = platform.node()
print("hostname:", hostname)
import parse as p
import sys
import glob

import classes_help 

if p.usecase == "u_hida":
    p.threshold = "100"
    p.use_unlabelled = 0
    
usecases_dict = {
    "u_hida": "2_fes_1_clf",
    "ss_hida": "2_fes_1_clf_semi"
    }
usecase = usecases_dict[p.usecase] 

num_class = p.num_class
use_unlabelled = p.use_unlabelled

# semisupervised
if "semi" in usecase:
    semisupervised = True
else:
    semisupervised = False

dataset = p.dataset

if dataset == 'space':
    if p.color_mode == 'rgb':
        color_mode_s, color_mode_t= 'rgb' , 'rgb'
    else:
        if p.source == 'resisc':
            color_mode_s, color_mode_t= 'rgb', 'multispectral' # multiband datasets
        else:
            color_mode_s, color_mode_t= 'multispectral' , 'rgb' # multiband datasets

#*************
processing_type = p.processing_type
print("processing_type: ", processing_type)

#*****************

data_root_dir = str(p.num_class) + '_classes'

augment = p.aug

print("data augmentation: ", augment)

########### parameters for imageDataGenerator and flow from directory;

#for imagedatagenerator object 

train_augmentation_args = dict(augmentationparameters= [augment])
valid_augmentation_args = dict(augmentationparameters=[False])

# for flow from directory method
train_augmentationclassblock={}
valid_augmentationclassblock={} 
##############

if use_unlabelled==1:
    train_wdgrl_adapter_separately=True #if set to true , domain critic will use all data target 
else:
    train_wdgrl_adapter_separately=False

with_critic = p.with_critic
train_wdgrl_adapter_separately = train_wdgrl_adapter_separately and with_critic

ms2rgb = False

repetition_no = None

count_epochs_source = True

staining_s = p.source
staining_t = p.target

domain_dir_s = staining_s
domain_dir_t = staining_t

    
classes = {}
    
# space
if dataset == 'space':
    if num_class <= 5:
        classes['eurosat'] = ['AnnualCrop',           'Forest',   'Industrial',      'Residential',        'River']
    else:
        classes['eurosat'] = ['Crop',           'Forest',   'Industrial',      'Residential',        'River',   "Highway",   "Pasture",  "SeaLake"]
    
    classes['resisc'] = ['rectangular_farmland',  'forest',  'industrial_area', 'dense_residential',   'river',  "freeway",   "meadow",   "lake"]

classes_s = classes[p.source]
classes_t = classes[p.target]
     
classes_s = classes_s[:num_class]
classes_t = classes_t[:num_class]   
        
if num_class == 0: #taking all subfolders
    classes_s = None
    classes_t = None

means = {p.source:0, p.target:0}
stddevs = {p.source:1, p.target:1}

if p.measure != None:
    measure = p.measure
elif usecase=='2_fes_1_clf':
    measure = 'cl_loss_s'
elif usecase=='2_fes_1_clf_semi':
    measure = 'cl_loss_t'
else:
    measure = ''
        
if measure == 'acc' or measure == 'f1':
    min_or_max='max'
else:
    min_or_max='min'


data_path = 'data/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base_outdir = data_path
logdirpath = os.path.join(base_outdir, 'target_' + p.target)
logdirpath += '_cases/target' + p.threshold
basepath = os.path.join(logdirpath, data_root_dir)
    
inputpath_s = os.path.join(basepath, domain_dir_s)
inputpath_t = os.path.join(basepath, domain_dir_t)

# Only for load_and_test.py    
def get_basepath(logdirpath):
    logdirpath += p.threshold
    basepath = os.path.join(logdirpath, data_root_dir)
    inputpath_s = os.path.join(basepath, domain_dir_s)
    inputpath_t = os.path.join(basepath, domain_dir_t)
    return basepath, inputpath_s, inputpath_t

