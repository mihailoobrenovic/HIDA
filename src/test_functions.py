# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import utils_da
import config as conf
import features_analysis as fa
import sys
import parse as p

from utils import help_functions
# Used for classifier and adapter, classification reslts for one set
def test_epoch_losses_one_set(model, sess, flow, domain, params,
                              gray2rgb=False, diff_inputs=False, source=False,
                              resize=None, ms2rgb=False):
    batch_size = params['batch_size']
    num_class = params['num_class']
    inp_shape_s = params['inp_shape_s']
    inp_shape_t = params['inp_shape_t']
    num_channels_s = params['num_channels_s']
    num_channels_t = params['num_channels_t']
    
    n_flow = flow.samples
    iters = n_flow // (batch_size // 2)
    # The last batch could be smaller than the others, we will use it
    if n_flow % (batch_size // 2) > 0:
        iters += 1
        
    loss_sum = 0
    conf_mat_sum = np.zeros((num_class, num_class))

    clf = model.clf

    eval_tensor_ops = [clf.total_loss, clf.conf_mat]
     
    for j in range(iters):
        x_val_batch, y_val_batch = next(flow)
        if ms2rgb:
            x_val_batch = utils_da.ms2rgb_set(x_val_batch)
        
        # Models with two inputs
        # When we test on source set, we fake data for target input.
        # When we test on target set, we fake data for source input.
        # put inp.flag to false as we are testing now
        # inp_2.flag is target flag, 
        # it denotes if this is testing on target data
        # put inp_2.flag to True, as that means we're testing target domain
        
        # Test on target set
        if source==False:
            inp1_shape = (batch_size//2, inp_shape_s[0], inp_shape_s[1], 
                          num_channels_s)
            results = sess.run(eval_tensor_ops, 
                    {model.inp.X: np.ones(inp1_shape, dtype=np.float32), 
                     model.inp.y_true: y_val_batch, model.inp.flag: False,
                     model.inp_2.X: x_val_batch, 
                     model.inp_2.y_true: y_val_batch, 
                     model.inp_2.flag: True})
        # Test on source set, put inp_2.flag (target flag) to False, 
        else:
            inp2_shape = (batch_size//2, inp_shape_t[0], inp_shape_t[1], 
                          num_channels_t)
            results = sess.run(eval_tensor_ops, 
                    {model.inp.X: x_val_batch, model.inp.y_true: y_val_batch, 
                     model.inp.flag: False, 
                     model.inp_2.X: np.ones(inp2_shape, dtype=np.float32), 
                     model.inp_2.y_true: y_val_batch, model.inp_2.flag: False})
        
        # Sum up all of the batch losses and batch confusion matrices
        batch_loss, batch_conf_mat = results[0], results[1]
        loss_sum += batch_loss
        conf_mat_sum += batch_conf_mat    
    
    total_loss = loss_sum / n_flow
    
    # For balanced datasets
    total_acc = np.trace(conf_mat_sum) / n_flow
    
    print(domain)
    print('classifier loss: %f, accuracy: %f' % (total_loss, total_acc))
    
    print('confusion matrix:')
    print(conf_mat_sum)
    
    set_losses = {'cl_loss': total_loss, 'acc': total_acc,
                  'conf_mat': conf_mat_sum}
    return set_losses



# Used for adapter, calculates Wasserstein loss and silhouette score
# on a pair of source and target sets
def test_epoch_common_losses(model, sess, flow_s, flow_t, domain, params, 
                             gray2rgb_s=False, gray2rgb_t=False, 
                             diff_inputs=False, include_silhouette=False):
    batch_size = params['batch_size']
    crit_inp_shape = params['crit_inp_shape']
    
    n_s = flow_s.samples
    n_t = flow_t.samples
    n_flow = min(n_s, n_t)
    
    iters = n_flow // (batch_size // 2)
    iters_s = n_s // (batch_size // 2)
    iters_t = n_t // (batch_size // 2)
    if n_s % (batch_size // 2) > 0:
        iters_s += 1
    if n_t % (batch_size // 2) > 0:
        iters_t += 1
    
    eval_tensor_ops = [model.crit.out_s_sum, model.crit.out_t_sum]
    if include_silhouette:
        eval_tensor_ops += [model.slicer.h_s, model.slicer.h_t]
        n_features = crit_inp_shape
        feat_s = np.zeros((0, n_features))
        feat_t = np.zeros((0, n_features))
    
    critic_s_total_sum = 0
    critic_t_total_sum = 0
    for j in range(iters):
        xs_batch, ys_batch = next(flow_s)
        xt_batch, yt_batch = next(flow_t)
        # Value of input 2 flag is not important here
        results = sess.run(eval_tensor_ops, 
                {model.inp.X: xs_batch, model.inp_2.X: xt_batch,
                 model.inp.flag: True, model.inp_2.flag: False})
        if not include_silhouette:
            [batch_critic_s, batch_critic_t] = results
        else:
            batch_critic_s = results[0]
            batch_critic_t = results[1]
            feat_s_j = results[2]
            feat_t_j = results[3]
            feat_s = np.concatenate((feat_s, feat_s_j))
            feat_t = np.concatenate((feat_t, feat_t_j))
        
        critic_s_total_sum += batch_critic_s
        critic_t_total_sum += batch_critic_t
        
    if (iters < iters_s):
        for j in range(iters, iters_s):
            next(flow_s)
    if (iters < iters_t):
        for j in range(iters, iters_t):
            next(flow_t)
        
    total_critic_s = critic_s_total_sum / n_flow
    total_critic_t = critic_t_total_sum / n_flow
    total_wd_loss = total_critic_s - total_critic_t
    
    print(domain)
    print('wd loss: %f' % (total_wd_loss))
    
    common_losses = {'wd_loss' : total_wd_loss}

    if include_silhouette:
        silh_score = fa.silhouette_domains(feat_s, feat_t)
        common_losses['silh_score'] = silh_score
        
    return common_losses








# Get train and validation losses on train and val set of one domain
# Used for classifier after each epoch
def test_one_set_train_and_val(model, sess, train_flow_test, valid_flow, 
                               losses_hist, params, gray2rgb=False, 
                               diff_inputs=False):
    t_losses = test_epoch_losses_one_set(
            model, sess, train_flow_test, 'train set', params, 
            gray2rgb=gray2rgb, diff_inputs=diff_inputs)
    v_losses = test_epoch_losses_one_set(
            model, sess, valid_flow, 'validation set', params, 
            gray2rgb=gray2rgb, diff_inputs=diff_inputs)
    t_losses['model_loss'] = t_losses[conf.measure]
    v_losses['model_loss'] = v_losses[conf.measure]
    
    losses_hist.append_epoch(t_losses, v_losses, min_or_max=conf.min_or_max)
    
    


    

# Get losses on val set of one domain
# Used for classifier and adapter
def test_one_set_val(model, sess, flow_test, losses_hist, params, 
                     gray2rgb=False, diff_inputs=False, source=False,
                     resize=None, ms2rgb=False):
    v_losses = test_epoch_losses_one_set(
            model, sess, flow_test, 'val set', params, gray2rgb=gray2rgb, 
            diff_inputs=diff_inputs, source=source, resize=resize,
            ms2rgb=ms2rgb)
    v_losses['model_loss'] = v_losses[conf.measure]
    losses_hist.append_epoch({}, v_losses, min_or_max=conf.min_or_max)
    

    

def merge_source_and_target_losses(source_losses, target_losses):
    source_losses = \
        dict((key+'_s', value) for (key, value) in source_losses.items())
    target_losses = \
        dict((key+'_t', value) for (key, value) in target_losses.items())
    merged_losses = {**source_losses, **target_losses}
    return merged_losses


def losses_one_set_val(
        model, sess, valid_flow_s, valid_flow_t, params, gray2rgb_source, 
        gray2rgb_target, diff_inputs):
    vs_losses = test_epoch_losses_one_set(
            model, sess, valid_flow_s, 'validation set source', params, 
            gray2rgb=gray2rgb_source, diff_inputs=diff_inputs, source=True)
    vt_losses = test_epoch_losses_one_set(
            model, sess, valid_flow_t, 'validation set target', params, 
            gray2rgb=gray2rgb_target, diff_inputs=diff_inputs, source=False)
    v_losses = merge_source_and_target_losses(vs_losses, vt_losses)
    return v_losses


# Get train and val losses for both source and target domains
# Used for adapters after each epoch
def test_two_sets_train_and_val(
        model, sess, train_flow_s_test, train_flow_t_test, valid_flow_s,
        valid_flow_t, losses_hist, params, test_wd=False, gray2rgb=False, 
        diff_inputs=False, include_silhouette=False):
    # ts_losses = test_epoch_losses_one_set(
    #         model, sess, train_flow_s_test, 'train set source', params, 
    #         gray2rgb=gray2rgb, diff_inputs=diff_inputs, source=True)
    # tt_losses = test_epoch_losses_one_set(
    #         model, sess, train_flow_t_test, 'train set target', params, 
    #         gray2rgb=gray2rgb, diff_inputs=diff_inputs, source=False)
    vs_losses = test_epoch_losses_one_set(
            model, sess, valid_flow_s, 'test set source', params, 
            gray2rgb=gray2rgb, diff_inputs=diff_inputs, source=True)
    vt_losses = test_epoch_losses_one_set(
            model, sess, valid_flow_t, 'test set target', params, 
            gray2rgb=gray2rgb, diff_inputs=diff_inputs, source=False)
    # t_losses = merge_source_and_target_losses(ts_losses, tt_losses)
    v_losses = merge_source_and_target_losses(vs_losses, vt_losses)
    if test_wd:
        common_losses = test_epoch_common_losses(
                model, sess, valid_flow_s, valid_flow_t, 'test set', 
                params, diff_inputs=diff_inputs, 
                include_silhouette=include_silhouette)
        v_losses = {**v_losses, **common_losses}
    
    if conf.measure == 'cl_loss':
        # t_losses['model_loss'] = t_losses['cl_loss_s'] + t_losses['cl_loss_t']
        v_losses['model_loss'] = v_losses['cl_loss_s'] + v_losses['cl_loss_t']
    else:
        # t_losses['model_loss'] = t_losses[conf.measure]
        v_losses['model_loss'] = v_losses[conf.measure]
    
    # losses_hist.append_epoch(t_losses, v_losses, min_or_max=conf.min_or_max)
    losses_hist.append_epoch({}, v_losses, min_or_max=conf.min_or_max)
    
    
def test_two_sets_val(
        model, sess, valid_flow_s, valid_flow_t, losses_hist, params, 
        test_wd=False, gray2rgb=False, diff_inputs=False, 
        include_silhouette=False):
    v_losses = losses_one_set_val(
        model, sess, valid_flow_s, valid_flow_t, params, gray2rgb, gray2rgb, 
        diff_inputs)
    if test_wd:
        common_losses = test_epoch_common_losses(
                model, sess, valid_flow_s, valid_flow_t, 'validation set', 
                params, diff_inputs=diff_inputs, 
                include_silhouette=include_silhouette)
        v_losses = {**v_losses, **common_losses}
    
    if conf.measure == 'cl_loss':
        v_losses['model_loss'] = v_losses['cl_loss_s'] + v_losses['cl_loss_t']
    else:
        v_losses['model_loss'] = v_losses[conf.measure]
    
    losses_hist.append_epoch({}, v_losses, min_or_max=conf.min_or_max)
    




