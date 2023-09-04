# -*- coding: utf-8 -*-
import tensorflow as tf
import math
import sys

def conv2d(input_tensor, num_input_channels, num_filters, filter_shape, 
           layer_name, strides=1, padding='SAME', act=tf.nn.relu, seed=None, 
           net_elem=None, batch_norm=False, is_train=False, init='wdgrl'):
    with tf.name_scope(layer_name):
        conv_filt_shape = [filter_shape[0], filter_shape[1], 
                           num_input_channels, num_filters]
        print(layer_name, input_tensor.shape, conv_filt_shape)
        if init == 'wdgrl':
            weight = tf.Variable(
                    tf.truncated_normal(
                            conv_filt_shape, stddev=0.03, seed=seed), 
                            name='weight')
            bias = tf.Variable(
                    tf.truncated_normal([num_filters], seed=seed), name='bias')
        elif init == 'dsn':
            fun_in = filter_shape[0] * filter_shape[1] * num_input_channels
            fun_out = filter_shape[0] * filter_shape[1] * num_filters
            n = (fun_in + fun_out) / 2.
            limit = math.sqrt(3.0 / n)
            weight = tf.Variable(
                    tf.random_uniform(conv_filt_shape, -limit, limit), 
                    name='weight')
            bias = tf.Variable(
                    tf.constant(0.0, shape=[num_filters]), name='bias')
        else:
            print("The init for that network is not defined")
            sys.exit()
        if net_elem != None:
            net_elem.W.append(weight)
            net_elem.b.append(bias)
        out_layer = tf.nn.conv2d(
                input_tensor, weight, strides=[1, strides, strides, 1], 
                padding=padding)
        out_layer = tf.nn.bias_add(out_layer, bias)
        if batch_norm:
            out_layer = tf.layers.batch_normalization(
                    out_layer, momentum=0.999, training=is_train)
        return act(out_layer)
    
    
def maxpool2d(input_tensor, k, padding='SAME'):
    ksize=[1, k, k, 1]
    strides=[1, k, k, 1]
    return tf.nn.max_pool(input_tensor, ksize=ksize, strides=strides, 
                          padding=padding)


def upsample(input_tensor, k):
    inp_shape = input_tensor.get_shape().as_list()
    inp_img_shape = (inp_shape[1], inp_shape[2])
    up_size = (round(k*inp_img_shape[0]), round(k*inp_img_shape[1]))
    return tf.image.resize_images(
            input_tensor, size=up_size, 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, 
             input_type='dense', seed=None, batch_norm=False, is_train=False,
             init='wdgrl', dropout=None):
    with tf.name_scope(layer_name):
        print(layer_name, input_tensor.shape, [input_dim, output_dim])
        if init == 'wdgrl':
            weight = tf.Variable(
                    tf.truncated_normal(
                            [input_dim, output_dim], 
                            stddev=1. / tf.sqrt(input_dim / 2.), seed=seed), 
                            name='weight')
            bias = tf.Variable(
                    tf.constant(0.1, shape=[output_dim]), name='bias')
        elif init == 'dsn':
            fun_in = input_dim
            fun_out = output_dim
            n = (fun_in + fun_out) / 2.
            limit = math.sqrt(3.0 / n)
            weight = tf.Variable(
                    tf.random_uniform([input_dim, output_dim], -limit, limit), 
                    name='weight')
            bias = tf.Variable(
                    tf.constant(0.0, shape=[output_dim]), name='bias')
        else:
            print("The init for that network is not defined")
            sys.exit()
            
        if input_type == 'sparse':
            multipl = tf.sparse_tensor_dense_matmul(input_tensor, weight)
            add_b = multipl + bias
            if batch_norm:
                add_b = tf.layers.batch_normalization(add_b, momentum=0.999, 
                                                      training=is_train)
            activations = act(add_b)
        else:
            multipl = tf.matmul(input_tensor, weight)
            add_b = multipl + bias
            if batch_norm:
                add_b = tf.layers.batch_normalization(add_b, momentum=0.999, 
                                                      training=is_train)
            activations = act(add_b)
            if dropout:
                activations = tf.nn.dropout(activations, rate=dropout)
        return activations, add_b, multipl