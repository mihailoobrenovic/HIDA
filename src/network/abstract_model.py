# -*- coding: utf-8 -*-
import tensorflow as tf
import network.network_utils as net_utils
import network.network_elements as net_elems
import math
import sys

class AbstractModel():
    
    
    def __init__(self):
        pass
    
    def add_input(self, inp_shape, num_channels, num_class, y_placeholder=True, 
                  input_scope_name='input'):
        with tf.name_scope(input_scope_name):
            inp = net_elems.Input(inp_shape, num_channels, num_class)
            inp.X = tf.placeholder(
                    dtype=tf.float32, 
                    shape=[None, inp_shape[0], inp_shape[1], num_channels])
            inp.flag = tf.placeholder(dtype=tf.bool)
            if y_placeholder:
                inp.y_true = tf.placeholder(dtype=tf.int32)
                inp.y_true_one_hot = tf.one_hot(inp.y_true, num_class)
        return inp
      
    
    def add_conv_part(self, component, layer_input, num_channels, 
                      filters_per_layer, conv_filter_shapes, maxpool_k, 
                      padding=None, maxpool_padding=None, batch_norm=False,
                      init='wdgrl'):
        n_conv_layers = len(filters_per_layer)
        n_inp_ch=num_channels
        # Convolutional layers
        for i in range(0, n_conv_layers):
            # print("conv", i, "batch norm", batch_norm[i])
            if padding==None:
                conv = net_utils.conv2d(layer_input, 
                                        num_input_channels=n_inp_ch, 
                                        num_filters=filters_per_layer[i], 
                                        filter_shape=conv_filter_shapes[i], 
                                        layer_name='conv'+str(i+1), 
                                        seed=self.seed, batch_norm=batch_norm[i],
                                        is_train=self.inp.flag,
                                        init=init, net_elem=component)
            else:
                conv = net_utils.conv2d(layer_input, 
                                        num_input_channels=n_inp_ch, 
                                        num_filters=filters_per_layer[i], 
                                        filter_shape=conv_filter_shapes[i], 
                                        layer_name='conv'+str(i+1), 
                                        seed=self.seed, padding=padding[i],
                                        batch_norm=batch_norm[i],
                                        is_train=self.inp.flag,
                                        init=init, net_elem=component)
            if maxpool_padding == None:
                pool = net_utils.maxpool2d(conv, k=maxpool_k[i])
            else:
                pool = net_utils.maxpool2d(conv, k=maxpool_k[i], 
                                           padding=maxpool_padding[i])
            layer_input = pool
            n_inp_ch = filters_per_layer[i]
            component.conv_layers.append({'conv':conv, 'pool':pool})
        return (component, layer_input)
    
    
    def flatten(self, component, layer_input, fc_inp_shape):
        component.flattened = tf.reshape(layer_input, [-1, fc_inp_shape])
        return component
    
    def unflatten(self, component, layer_input, conv_inp_shape):
        component.unflattened = tf.reshape(layer_input, 
                                           [-1, conv_inp_shape[0],
                                            conv_inp_shape[1],
                                            conv_inp_shape[2]])
        return component
            
    
    def add_fc_part(self, component, layer_input, fc_inp_shape, 
                    nodes_per_layer, batch_norm=False, init='wdgrl',
                    dropout=None):
        n_fc_layers = len(nodes_per_layer)
        layer_input_shape = fc_inp_shape
        for i in range(0, n_fc_layers):
            # print("fc", i, "batch norm", batch_norm[i])
            if dropout and dropout[i]:
                fc, add_b, multipl  = net_utils.fc_layer(
                        layer_input, layer_input_shape, nodes_per_layer[i], 
                        layer_name='dense_h'+str(i+1), seed=self.seed, 
                        batch_norm=batch_norm[i], is_train=self.inp.train_flag,
                        init=init, dropout=dropout[i])
            else:
                fc, add_b, multipl  = net_utils.fc_layer(
                        layer_input, layer_input_shape, nodes_per_layer[i], 
                        layer_name='dense_h'+str(i+1), seed=self.seed, 
                        batch_norm=batch_norm[i], is_train=self.inp.train_flag,
                        init=init)
            layer_input = fc
            layer_input_shape = nodes_per_layer[i]
            component.fc_layers.append(fc)
            component.add_bs.append(add_b)
            component.multipls.append(multipl)
        return (component, layer_input)
    
    
    def add_component(self, component, X, num_channels, conv_filter_shapes, 
                      filters_per_layer, padding, maxpool_k, 
                      maxpool_padding=None, nodes_per_layer=[], 
                      scope_name='component', batch_norm=False, init='wdgrl',
                      dropout=None):
        num_conv = len(conv_filter_shapes)
        num_fc = len(nodes_per_layer)
        if type(batch_norm) == list:
            num_layers = len(batch_norm)
            msg = "Length of batch_norm and number of layers not compatible"
            assert num_layers == num_conv + num_fc, msg
            batch_norm_conv = batch_norm[:num_conv]
            batch_norm_fc = batch_norm[num_conv:num_layers]
        else:
            batch_norm_conv = [batch_norm]*num_conv
            batch_norm_fc = [batch_norm]*num_fc
        with tf.name_scope(scope_name):
            layer_input = X
            (component, layer_input) = self.add_conv_part(component, layer_input, 
                                                    num_channels, 
                                                    filters_per_layer, 
                                                    conv_filter_shapes, 
                                                    maxpool_k, padding, 
                                                    maxpool_padding,
                                                    batch_norm=batch_norm_conv,
                                                    init=init)
            
            if nodes_per_layer != []:
                # Flatten
                fc_inp_shape = layer_input.shape[1].value \
                                * layer_input.shape[2].value \
                                * layer_input.shape[3].value
                component = self.flatten(component, layer_input, fc_inp_shape)
                # Fully connected layers
                layer_input = component.flattened
                (component, layer_input) = self.add_fc_part(
                        component, layer_input, fc_inp_shape, nodes_per_layer, 
                        batch_norm=batch_norm_fc,init=init, dropout=dropout)
            component.output = layer_input
        return component
   
    
    def add_basic_cond(self, cond, opt1, opt2):
        return tf.cond(cond, lambda: opt1, lambda: opt2)
        
    
    def add_slice_data(self, h, y, slice_flag, batch_size=0, 
                       slice_scope_name='slice_data'):
        with tf.name_scope(slice_scope_name):
            
            len_s = tf.shape(h)[0]//2
            len_t = tf.shape(h)[0] - len_s
            num_or_size_splits = tf.stack([len_s, len_t])
            
            h_s, h_t = tf.split(h, num_or_size_splits=num_or_size_splits, axis=0)
            
            sl = net_elems.Slicer()
            sl.h_s = tf.cond(slice_flag, lambda: h_s, lambda: h)
            sl.h_t = tf.cond(slice_flag, lambda: h_t, lambda: h)
            
            if y != None:
                ys_true, yt_true = tf.split(y, num_or_size_splits=num_or_size_splits, axis=0)
                sl.ys_true = tf.cond(slice_flag, lambda: ys_true, lambda: y)
                sl.yt_true = tf.cond(slice_flag, lambda: yt_true, lambda: y)
            return sl

    
    def add_splitter(self, h, train_flag, batch_size, 
                     split_scope_name='splitter'):
        return self.add_slice_data(h, None, train_flag, batch_size, 
                                   slice_scope_name=split_scope_name)
    
    
    def add_classifier(self, h, y_one_hot, y_label, num_nodes, num_class, 
                       test_results=True, cl_scope_name='classifier',
                       nodes_per_layer=[], batch_norm=False, init='wdgrl'):  
        with tf.name_scope(cl_scope_name):
            clf = net_elems.Classifier()
            print(cl_scope_name, h.shape, [num_nodes, num_class])
            
            if nodes_per_layer != []:
                layer_input = h
                fc_inp_shape = num_nodes
                (clf, h) = self.add_fc_part(clf, layer_input, fc_inp_shape, 
                                       nodes_per_layer, batch_norm=batch_norm,
                                       init=init)
                num_nodes = nodes_per_layer[-1]
            
            # Last classifying layer
            if init == 'wdgrl':
                clf.W = tf.Variable(
                        tf.truncated_normal(
                                [num_nodes, num_class], 
                                stddev=1. / tf.sqrt(num_nodes / 2.), 
                                seed=self.seed), name='clf_weight')
                clf.b = tf.Variable(
                        tf.constant(0.1, shape=[num_class]), name='clf_bias')
            else:
                print("The init for that network is not defined")
                sys.exit()
            self.cl_softmax_layer(clf, h, clf.W, clf.b, y_one_hot, y_label, test_results, num_class)
            return clf
        
    def cl_softmax_layer(self, clf, h, W, b, y_one_hot, y_label, test_results, 
                         num_class, weight_loss=1):
        clf.pred_logit = tf.matmul(h, W) + b
        clf.pred_softmax = tf.nn.softmax(clf.pred_logit)
        clf.y_pred = tf.argmax(clf.pred_softmax, 1)
        clf.loss = weight_loss * tf.losses.softmax_cross_entropy(
                logits=clf.pred_logit, onehot_labels=y_one_hot)
        if test_results:
                self.add_classifier_tests(clf, y_one_hot, y_label, num_class)
        
    def add_classifier_tests(self, clf, y_one_hot, y_label, num_class):

        clf.total_loss = tf.losses.softmax_cross_entropy(
                logits=clf.pred_logit, onehot_labels=y_one_hot,
                reduction=tf.losses.Reduction.SUM)
        # Confusion matrix function requires label-encoding instead of one-hot
        decoded_labels = tf.argmax(y_one_hot, axis=1)
        clf.conf_mat = tf.confusion_matrix(decoded_labels, clf.y_pred, 
                                           num_classes=num_class)
    
    
    def vstack_s_and_t(self, t_s, t_t):
        return tf.concat([t_s, t_t], axis=0)
    
    
    def hstack_s_and_t(self, t_s, t_t):
        return tf.concat([t_s, t_t], axis=1)
        
    
    def add_wd_test(self, crit):
        # For testing wd loss on whole set
        crit.out_s_sum = tf.reduce_sum(crit.out_s)
        crit.out_t_sum = tf.reduce_sum(crit.out_t)
        
    
    def add_tensorboard(self):
        tb = net_elems.Tensorboard()
        self.tb = tb
    
