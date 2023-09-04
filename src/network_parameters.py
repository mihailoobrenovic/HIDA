# -*- coding: utf-8 -*-
import config as conf

import parse as p

import copy

num_class = p.num_class
usecase = p.usecase
source = p.source
target = p.target
if p.batch_norm == 1:
    batch_norm = True
else:
    batch_norm = False
    
architecture = p.architecture

def get_arch_256(num_channels):
    arch_256 = {
        'inp_shape' : (256, 256),
        'num_channels' : num_channels,
        'maxpool_k_enc' : [4, 4],
        'upsample_k' : [2, 4, 4],
        'filters_per_layer_dec' : [32, 16, num_channels]
    }
    return arch_256

def get_arch_64(num_channels):
    arch_64 = {
        'inp_shape' : (64, 64),
        'num_channels' : num_channels,
        'maxpool_k_enc' : [2, 2],
        'upsample_k' : [2, 2, 2],
        'filters_per_layer_dec' : [32, 16, num_channels]
    }
    return arch_64

rgb_channels = 3
ms_channels = 13

arch = {}

# Remote sensing
arch['resisc'] = get_arch_256(rgb_channels)
if p.color_mode == "rgb":
    arch['eurosat'] = get_arch_64(rgb_channels) #  for eurosat rgb
else:
    arch['eurosat'] = get_arch_64(ms_channels) #  for eurosat multispectral


params_space = {
    'inp_shape_s' : arch[p.source]['inp_shape'],
    'inp_shape_t' : arch[p.target]['inp_shape'],
    'num_channels_s' : arch[p.source]['num_channels'],
    'num_channels_t' : arch[p.target]['num_channels'],
    'num_class' : num_class,
    
    'conv_filter_shapes_enc' : [[5,5], [5,5], [5,5]],
    'filters_per_layer_enc' : [16, 32, 32],
    'padding_enc_s' : ['SAME', 'SAME', 'SAME'],
    'padding_enc_t' : ['SAME', 'SAME'],
    'maxpool_k_enc' : arch[p.source]['maxpool_k_enc']+[2],
    'maxpool_k_enc_t' : arch[p.target]['maxpool_k_enc'],
    
    'fc_inp_shape_enc' : None,
    'nodes_per_layer_enc' : [],
    
    'upsample_k_s' : arch[p.source]['upsample_k'],
    'upsample_k_t' : arch[p.target]['upsample_k'],
    'conv_filter_shapes_dec' : [[5,5], [5,5], [5,5]],
    'filters_per_layer_dec_s' : arch[p.source]['filters_per_layer_dec'],
    'filters_per_layer_dec_t' : arch[p.target]['filters_per_layer_dec'],

    'padding_dec_s' : ['SAME', 'SAME', 'SAME'],
    'padding_dec_t' : ['SAME', 'SAME', 'SAME'],

    'num_common_layers_enc' : 1,
    
    # Convolutional part of feature extractor, case with AEs
    'conv_filter_shapes' : [[5,5]],
    'filters_per_layer' : [64],
    'padding' : ['SAME'],
    'maxpool_k' : [2],
    
    # Fully connected part of feature extractor
    'fc_inp_shape' : 4*4*64,
    'fe_nodes_per_layer' : [100],
    
    # Domain critic
    'crit_inp_shape' : 100,
    'crit_nodes_per_layer' : [100],
    
    'batch_size' : p.batch_size,
    'batch_size_feat_analysis' : 1120,
   
    # For segmentation
    # 'otp_shape' : (64, 64)
    'otp_shape' : (256, 256)
}


params = params_space

params['seed'] = None
params['maxpool_padding'] = ['SAME', 'SAME']

params['conv_filter_shapes'] = \
    params['conv_filter_shapes_enc'] + params['conv_filter_shapes']
params['filters_per_layer'] = \
    params['filters_per_layer_enc'] + params['filters_per_layer']
params['padding'] = params['padding_enc_s'] + params['padding']
params['maxpool_k'] = params['maxpool_k_enc'] + params['maxpool_k']
        
params['num_separated_layers_s'] = \
    len(params['maxpool_k_enc']) - params['num_common_layers_enc']
params['num_separated_layers_t'] = len(params['maxpool_k_enc_t'])
params['num_common_layers'] = \
    params['num_common_layers_enc'] + len(params['filters_per_layer'])
params['padding_t'] = params['padding_enc_t']
params['maxpool_k_t'] = params['maxpool_k_enc_t']

if num_class == 8:
    params['num_steps'] = 8000
else:
    params['num_steps'] = 5000

params['num_epochs'] = p.epochs

params['lr_wd_D'] = 1e-3  # Learning rate for training the Wasserstein distance domain critic
params['gp_param'] = 10   # Gradient penalty regularization parameter when training the domain critic
params['l2_param'] = 1e-5 # L2 regularization parameter when training the feature extractor and the classifier
params['wd_param'] = 0.1  # The weight parameter of the Wasserstein distance loss in the total loss equation in train_adapter
params['lr'] = 1e-4
params['optimizer'] = 'adam'

params['D_train_num'] = 10  # Number of iterations to train domain critic in one training step






########## params batch norm
params["batch_norm_fe"]        = [batch_norm, batch_norm] 
params["batch_norm_fe2"]       = [batch_norm, batch_norm]
params["batch_norm_fe_common"] = [False, False, False]



params["batch_norm_clf_s"]      = False         # should be set to False
params["batch_norm_clf_t"]      = False         # should be set to False
params["batch_norm_clf_common"] = False        #for usecase == '2_fes_1_clf_semi';  should be set to False


params["batch_norm_domain_critic"] = False 






















