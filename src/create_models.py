# -*- coding: utf-8 -*-
import network.model as net_model

import config as conf
import parse as p



def create_wdgrl_adapter_with_two_fes(params, test_clf=True, test_crit=False, 
                                   tb=False):
    m = net_model.Model(params['seed'])
    m.inp = m.add_input(params['inp_shape_s'], params['num_channels_s'], 
                        params['num_class'], input_scope_name='input')
    m.inp_2 = m.add_input(params['inp_shape_t'], params['num_channels_t'], 
                          params['num_class'], input_scope_name='input_2')
    num_separated_layers_s = params['num_separated_layers_s']
    num_separated_layers_t = params['num_separated_layers_t']
    
    m.fe = m.add_feature_extractor(
            m.inp.X, m.inp.num_channels, 
            params['conv_filter_shapes'][:num_separated_layers_s], 
            params['filters_per_layer'][:num_separated_layers_s], 
            params['maxpool_k'][:num_separated_layers_s],
            fc_inp_shape=params['fc_inp_shape_enc'],
            nodes_per_layer=params['nodes_per_layer_enc'],
            fe_scope_name='feature_extractor', 
            padding=params['padding'][:num_separated_layers_s], batch_norm=params["batch_norm_fe"]) 
    m.fe_2 = m.add_feature_extractor(
            m.inp_2.X, m.inp_2.num_channels, 
            params['conv_filter_shapes'][:num_separated_layers_t], 
            params['filters_per_layer'][:num_separated_layers_t], 
            params['maxpool_k_t'], fc_inp_shape=params['fc_inp_shape_enc'],
            nodes_per_layer=params['nodes_per_layer_enc'], 
            fe_scope_name='feature_extractor_2',
            padding=params['padding_t'][:num_separated_layers_t], batch_norm=params["batch_norm_fe2"])          
    
    m.merge = m.add_merger(m.fe.output, m.inp.y_true_one_hot, m.inp.y_true,
                           m.fe_2.output, m.inp_2.y_true_one_hot, 
                           m.inp_2.y_true, m.inp.flag, m.inp_2.flag)
    
    m.fe_common = m.add_feature_extractor(
            m.merge.h, m.merge.h.shape[-1].value, 
            params['conv_filter_shapes'][num_separated_layers_s:],
            params['filters_per_layer'][num_separated_layers_s:], 
            params['maxpool_k'][num_separated_layers_s:], 
            nodes_per_layer=params['fe_nodes_per_layer'],
            fe_scope_name='feature_extractor_common',
            padding=params['padding'][num_separated_layers_s:], batch_norm=params["batch_norm_fe_common"])   ####################pass batch norm

    
    m.slicer = m.add_slice_data(m.fe_common.output, m.merge.y_one_hot, 
                                m.inp.flag, params['batch_size'])
    
    clf_t = None
    if len(params['fe_nodes_per_layer']) > 0:
        clf_fc_inp_shape = params['fe_nodes_per_layer'][-1]
    else:
        clf_fc_inp_shape = params['nodes_per_layer_enc'][-1]
    
    if conf.usecase == '2_fes_1_clf_semi':
        m.clf = m.add_classifier(
                m.fe_common.output, m.merge.y_one_hot, m.merge.y_label, 
                clf_fc_inp_shape, params['num_class'], 
                test_results=test_clf, cl_scope_name='classifier', 
                batch_norm=params["batch_norm_clf_common"])
        
    else:   # 2_fes_1_clf (U-HIDA)
        m.clf = m.add_classifier(m.slicer.h_s, m.slicer.ys_true, m.merge.y_label, 
                             clf_fc_inp_shape, params['num_class'], 
                             test_results=test_clf, cl_scope_name='classifier', batch_norm=params["batch_norm_clf_s"])   
    
    if conf.with_critic:
        m.crit = m.add_critic(m.fe_common.output, m.slicer.h_s, m.slicer.h_t, 
                              params['crit_inp_shape'], 
                              params['crit_nodes_per_layer'], m.inp.flag, 
                              params['batch_size'], test_results=test_crit, batch_norm=params["batch_norm_domain_critic"])   
        
        m.train = m.train_adapter(m.clf, m.crit, params['lr_wd_D'], 
                                  params['gp_param'], params['l2_param'], 
                                  params['wd_param'], params['lr'], clf_t=clf_t)
    else:
        m.crit = None  
        
        m.train = m.train_adapter_without_critic(m.clf, params['l2_param'], 
                                                 params['lr'], clf_t=clf_t)        
        

    if tb:
        m.add_tensorboard()
    return m





