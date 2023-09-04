# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf
import os
import time
import numpy as np
import sys
import config as conf
import create_models as create
import generators as gen
from history import History
import test_functions as test
import utils_da
import parse as p


 
    
def str_date_time(datetime):
    return '{}-{}-{}_{}-{}-{}'.format(datetime.year, datetime.month, datetime.day, 
            datetime.hour, datetime.minute, datetime.second)


def set_paths_for_train_two_sets(model_name, color_mode_s, color_mode_t):
    today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    domain_s = conf.staining_s
    domain_t = conf.staining_t
    pathname = model_name+'-'+domain_s+'-'
    if domain_s == domain_t:
        pathname += color_mode_s+'-'+color_mode_t+'-'
    else:
        pathname += domain_t+'-'+p.color_mode+'-'
    if p.batch_norm:
        pathname += 'batch_norm-'
    if p.num_class:
        pathname += str(p.num_class) + '_classes-'
    
    # Make dir structure - .../exp_name/
    folder_name = pathname[:-1]
    if p.exp_name:
        folder_name = os.path.join(folder_name, p.exp_name)
    if p.threshold:
        folder_name = os.path.join(folder_name, p.threshold)    
    
    # Make pickle/checkpoint name 
    pathname = model_name+'-'

    pathname += today
    
    checkpoint_folder = os.path.join('checkpoints', folder_name)
    pickle_folder = os.path.join('pickles', folder_name)
    
    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs(pickle_folder, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_folder, pathname)
    pickle_path = os.path.join(pickle_folder, pathname+'.pkl')
    
    print("Checkpoints saved in: ", checkpoint_path)
    return checkpoint_path, pickle_path


def setup_saving_for_train(checkpoint_on, pickle_path):
    if checkpoint_on == 0:
        max_to_keep = None
    else:
        max_to_keep = 5
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    losses_hist = History(path=pickle_path)
    return saver, losses_hist


def get_batch_for_train(train_flow, batch_size):
    x_batch, y_batch = next(train_flow)
    # Skipping the last batch in epoch if it's smaller than the batch size
    if (x_batch.shape[0] < batch_size // 2):
        x_batch, y_batch = next(train_flow)
    return x_batch, y_batch


def get_batches_for_train(train_flow_s, train_flow_t, batch_size):
     
    xs_batch, ys_batch = get_batch_for_train(train_flow_s, batch_size)
    xt_batch, yt_batch = get_batch_for_train(train_flow_t, batch_size)
    # Skipping the last batch in epoch if it's smaller than the batch size
    return xs_batch, ys_batch, xt_batch, yt_batch


def prepare_session_two_domains(
        sess, train_flow_s, mean_s, mean_t, stddev_s, stddev_t, params, 
        color_mode_s, color_mode_t, tb, validate_all, validate_one, model):
    n_s = train_flow_s.samples    
    epoch_iters = n_s // (params['batch_size'] // 2)
    
    train_flow_s_test = train_flow_t_test = valid_flow_s = None
    if validate_all == True:
        # train_flow_s_test, train_flow_t_test, valid_flow_s, valid_flow_t = \
        valid_flow_s, valid_flow_t = \
            gen.get_all_generators(
                    params['inp_shape_s'], params['inp_shape_t'], 
                    params['otp_shape'], params['num_class'], 
                    params['batch_size'], color_mode_s, color_mode_t, train_test=True)
    elif validate_one == True:
        valid_flow_t = gen.get_val_generator(
                params['inp_shape_t'], params['otp_shape'], 
                params['num_class'], params['batch_size'], color_mode_t, 
                source=False, mean=mean_t, stddev=stddev_t)  
    if tb:
        model.tb.writer = tf.summary.FileWriter(
                os.path.join('output', p.exp_name, str(int(time.time()))), sess.graph)
        
    print("Begin")
    epoch_cnt = 1
    return epoch_iters, train_flow_s_test, train_flow_t_test, valid_flow_s, \
        valid_flow_t, epoch_cnt
    

def train_adapter(params, validate_all=True, validate_one=False, save_on=50, 
                  checkpoint_on=1000, test_wd=False,  color_mode_s='rgb', 
                  color_mode_t='rgb', tb=False, continue_training=False, 
                  restore_checkpoint=None, include_silhouette=False,
                  two_fes=False):
    model_name = p.usecase
    
    checkpoint_path, pickle_path = \
        set_paths_for_train_two_sets(model_name, color_mode_s, color_mode_t)
    
    test_wd = test_wd and conf.with_critic
    
    if continue_training == True:
        restore_checkpoint_path = os.path.join('checkpoints', 
                                               restore_checkpoint)

    model = create.create_wdgrl_adapter_with_two_fes(
        params, test_clf=True, test_crit=test_wd, tb=tb)
    
    if not conf.train_wdgrl_adapter_separately:
        train_flow_s, train_flow_t, mean_s, mean_t, stddev_s, stddev_t = \
            gen.get_train_generators(
                    params['inp_shape_s'], params['inp_shape_t'], 
                    params['otp_shape'], params['num_class'], 
                    params['batch_size'], color_mode_s, color_mode_t)
            
        wdgrl_train_flow_t, wdgrl_mean_t, wdgrl_stddev_t = None, None, None
        
        
    elif conf.train_wdgrl_adapter_separately:
        train_flow_s, train_flow_t, wdgrl_train_flow_t, mean_s, mean_t, wdgrl_mean_t, stddev_s, stddev_t, wdgrl_stddev_t = \
            gen.get_train_generators(
                    params['inp_shape_s'], params['inp_shape_t'], 
                    params['otp_shape'], params['num_class'], 
                    params['batch_size'], color_mode_s, color_mode_t, wdgrl_train_flow=conf.train_wdgrl_adapter_separately)

       
    with tf.Session() as sess:
        saver, losses_hist = setup_saving_for_train(checkpoint_on, pickle_path)
   
        if continue_training == False:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, restore_checkpoint_path)

        epoch_iters, train_flow_s_test, train_flow_t_test, valid_flow_s, \
            valid_flow_t, epoch_cnt = \
            prepare_session_two_domains(
                sess, train_flow_s, mean_s, mean_t, stddev_s, stddev_t, params,
                color_mode_s, color_mode_t, tb, validate_all, validate_one, 
                model)
            
        if not conf.count_epochs_source:
            # If we count epochs per target data 
            n_t = train_flow_t.samples    
            epoch_iters = n_t // (params['batch_size'] // 2)
            
        losses_hist.set_steps_per_epoch(epoch_iters)
        
        num_steps = params['num_epochs'] * epoch_iters

        if continue_training:
            i_start = sess.run(model.train.global_step)
        else:
            i_start = 0
        
        for i in range(i_start, num_steps):
            # Batches
            xs_batch, ys_batch, xt_batch, yt_batch = \
                get_batches_for_train(
                    train_flow_s, train_flow_t, params['batch_size'])
            if conf.train_wdgrl_adapter_separately:
                xt_batch_wdgrl, yt_batch_wdgrl = get_batch_for_train(wdgrl_train_flow_t, params['batch_size'])
                
            # Case of 2 FEs
            # Input 1 flag is true - train phase
            # Input 2 flag is not used here - can be any value            
            # Feed dicts
            
            feed_dict_cl = {
                    model.inp.X: xs_batch, model.inp.y_true: ys_batch, 
                    model.inp.flag: True, model.inp_2.X: xt_batch, 
                    model.inp_2.y_true: yt_batch, model.inp_2.flag: True}
            
            
            if not conf.train_wdgrl_adapter_separately:
                 feed_dict_crit = {model.inp.X: xs_batch, model.inp.flag: True, 
                              model.inp_2.X: xt_batch, 
                              model.inp_2.flag: True}
                 
                 
            elif conf.train_wdgrl_adapter_separately:
                feed_dict_crit = {model.inp.X: xs_batch, model.inp.flag: True, 
                              model.inp_2.X: xt_batch_wdgrl, 
                              model.inp_2.flag: True}
                
            # Train domain critic
            wl = 0
            if conf.with_critic:
                for j in range(params['D_train_num']):
                    _, wl = sess.run([model.train.wd_d_op, model.crit.wd_loss], 
                                     feed_dict=feed_dict_crit)
                if p.wd_loss_on == 'unsupervised':
                    if conf.train_wdgrl_adapter_separately:
                        _, wl = sess.run([model.train.wd_fe_op, model.crit.wd_loss], 
                                          feed_dict=feed_dict_crit)

            run_ops_cl = [model.train.train_op, model.train.total_loss, 
                          model.clf.loss]
            _, tl, cl = sess.run(run_ops_cl, feed_dict=feed_dict_cl)
            
            # Save losses to pickle

            if (i+1)%save_on==0:
                losses_dict = {'tl': tl, 'cl': cl, 'wl': wl}
                losses_hist.append_step(losses_dict)  
                
            if (i+1)%epoch_iters==0:
                print("Step", i+1, "epoch", epoch_cnt)
                epoch_cnt += 1
                diff_inputs = True
                
                # If we test on all sets
                if validate_all:
                    test.test_two_sets_train_and_val(
                            model, sess, train_flow_s_test, train_flow_t_test, 
                            valid_flow_s, valid_flow_t, losses_hist, params,
                            test_wd=test_wd, diff_inputs=diff_inputs,
                            include_silhouette=include_silhouette)
                # If we test only on one set
                elif validate_one:
                    test.test_one_set_val(
                            model, sess, valid_flow_t, losses_hist, params,
                            diff_inputs=diff_inputs)
                # Either we test on all sets or one set, save the losses and do checkpoint if needed
                if validate_all or validate_one or test_wd:
                    losses_hist.save_epoch()
                    if not p.last_chckp_only:
                        if checkpoint_on == 0:
                            losses_hist.checkpoint = True
                        if losses_hist.checkpoint:
                            saver.save(sess, checkpoint_path, 
                                       global_step=model.train.global_step)
            if not p.last_chckp_only:
                if checkpoint_on != 0 and (i+1)%checkpoint_on==0:
                    saver.save(sess, checkpoint_path, 
                               global_step=model.train.global_step)
                    
        if p.last_chckp_only:
            saver.save(sess, checkpoint_path, global_step=model.train.global_step)
                





    
    


       

