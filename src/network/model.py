# -*- coding: utf-8 -*-
import tensorflow as tf
import network.network_utils as net_utils
import network.network_elements as net_elems
from network.abstract_model import AbstractModel

import parse as p
  
      
class Training():
    def __init__(self):
        # global step for checkpoints
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, 
                                       name='global_step')
        self.gradients = None
        self.slopes = None
        self.gradient_penalty = None
        self.theta_C = None
        self.theta_D = None
        self.theta_G = None
        self.wd_d_op = None
        self.variables = None
        self.l2_loss = None
        self.total_loss = None
        self.optimizer = None
        self.grads = None
        self.train_op = None
        self.theta_C_1 = None
        self.theta_G_1 = None
        self.theta_C_2 = None
        self.theta_G_2 = None
        self.variables_adapt = None
        self.variables_clf = None
        self.l2_loss_adapt = None
        self.l2_loss_clf = None
        self.total_loss_adapt = None
        self.total_loss_clf = None
        self.optimizer_adapt = None
        self.optimizer_clf = None
        self.grads_adapt = None
        self.grads_clf = None
        self.train_op_adapt = None
        self.train_op_clf = None
        


class Model(AbstractModel):
    
    def __init__(self, seed=None):
        tf.reset_default_graph()
        tf.set_random_seed(seed)    
        self.seed = seed
        self.inp = None
        self.fe = None
        self.slicer = None
        self.clf = None
        self.crit = None
        self.train = None
        # For two different generators
        self.inp_2 = None
        self.fe_2 = None
        self.condit = None
        self.merge = None
        # For Tensorboard
        self.tb = None
        # When testing adapter and classifier at the same time
        # FE 2 already exists
        self.slicer_2 = None
        self.clf_2 = None
        self.train_adapt_and_clf = None
        
    
    def add_feature_extractor(self, X, num_channels, conv_filter_shapes, 
                      filters_per_layer, maxpool_k, fc_inp_shape=None, 
                      nodes_per_layer=[], fe_scope_name='feature_extractor',
                      padding=None, batch_norm=False, dropout=None):
        fe = net_elems.FeatureExtractor()
        fe = self.add_component(
                fe, X, num_channels, conv_filter_shapes, filters_per_layer, 
                padding, maxpool_k, nodes_per_layer=nodes_per_layer, 
                scope_name=fe_scope_name, batch_norm=batch_norm,
                dropout=dropout) ###add batch norm
        return fe
        
        
    def add_condition(self, h_s, y_one_hot_s, y_label_s, h_t, y_one_hot_t, 
                      y_label_t, train_flag, cond_scope_name='condition'):
        with tf.name_scope(cond_scope_name):
            condit = net_elems.Condition()
            condit.h = tf.cond(train_flag, lambda: h_s, lambda: h_t)
            condit.y_one_hot = tf.cond(train_flag, lambda: y_one_hot_s, 
                                       lambda: y_one_hot_t)
            condit.y_label = tf.cond(train_flag, lambda: y_label_s, 
                                     lambda: y_label_t)
            return condit
        
        
    def add_merger(self, h_s, y_one_hot_s, y_label_s, h_t, y_one_hot_t, 
                   y_label_t, train_flag, target_flag, 
                   merge_scope_name='merger'):
        with tf.name_scope(merge_scope_name):
            merge = net_elems.Merger()
            merge.h = tf.cond(
                    train_flag, lambda: self.vstack_s_and_t(h_s, h_t),
                    lambda: tf.cond(target_flag, lambda: h_t, lambda: h_s))
            merge.y_one_hot = tf.cond(
                    train_flag, 
                    lambda: self.vstack_s_and_t(y_one_hot_s, y_one_hot_t), 
                    lambda: tf.cond(
                            target_flag, lambda: y_one_hot_t, 
                            lambda: y_one_hot_s))
            merge.y_label = tf.cond(
                    train_flag, 
                    lambda: self.vstack_s_and_t(y_label_s, y_label_t), 
                    lambda: tf.cond(
                            target_flag, lambda: y_label_t, lambda: y_label_s))
            return merge
    
    
    def add_critic(self, h, h_s, h_t, crit_inp_shape, nodes_per_layer, 
                   slice_flag, batch_size, test_results=False, 
                   crit_scope_name='critic', batch_norm=False):
        crit = net_elems.Critic()
        batch_size = tf.shape(h)[0]
        crit.alpha = \
            tf.random_uniform(
                    shape=[batch_size // 2, 1], minval=0., maxval=1., 
                    seed=self.seed)
        crit.differences = h_s - h_t
        crit.interpolates = h_t + (crit.alpha*crit.differences)
        crit.h_whole = tf.concat([h, crit.interpolates], 0)
        
        with tf.name_scope(crit_scope_name):
            n_fc_layers = len(nodes_per_layer)
            layer_input = crit.h_whole
            layer_input_shape = crit_inp_shape
            for i in range(0, n_fc_layers):
                fc, _, _ = net_utils.fc_layer(
                        layer_input, layer_input_shape, nodes_per_layer[i], 
                        layer_name='critic_h'+str(i+1), seed=self.seed, batch_norm=batch_norm)  # pass batch norm
                layer_input = fc
                layer_input_shape = nodes_per_layer[i]
                crit.fc_layers.append(fc)
            crit.output, _, _ = net_utils.fc_layer(
                    layer_input, layer_input_shape, 1, 
                    layer_name='critic_h'+str(n_fc_layers+1), act=tf.identity, 
                    seed=self.seed, batch_norm=batch_norm)
            
        crit.out_s = tf.cond(
                slice_flag, 
                lambda: tf.slice(crit.output, [0, 0], [batch_size // 2, -1]), 
                lambda: crit.output)
        crit.out_t = tf.cond(
                slice_flag, 
                lambda: tf.slice(
                        crit.output, 
                        [batch_size // 2, 0], [batch_size // 2, -1]), 
                lambda: crit.output)
        crit.wd_loss = \
            (tf.reduce_mean(crit.out_s) - tf.reduce_mean(crit.out_t))
        # Now for the testing
        if test_results:
            self.add_wd_test(crit)
        return crit    
    
    
    def train_adapter(self, clf, crit, lr_wd_D, gp_param, l2_param, wd_param, 
                      lr, clf_t=None):
        train = Training()
        train.gradients = tf.gradients(crit.output, [crit.h_whole])[0]
        train.slopes = \
            tf.sqrt(tf.reduce_sum(
                            tf.square(train.gradients), reduction_indices=[1]))
        train.gradient_penalty = tf.reduce_mean((train.slopes-1.)**2)
        train.theta_C = \
            [v for v in tf.trainable_variables() if 'classifier' in v.name]
        train.theta_D = \
            [v for v in tf.trainable_variables() if 'critic' in v.name]
        train.theta_G = \
            [v for v in tf.trainable_variables() 
            if 'feature_extractor' in v.name]
        
        # Training domain critic with wd loss
        train.wd_d_op = \
            tf.train.AdamOptimizer(lr_wd_D).minimize(
                    -crit.wd_loss+gp_param*train.gradient_penalty, 
                    var_list=train.theta_D)
            
        if p.wd_loss_on == 'unsupervised':
            # Training FE with wd_loss
            train.wd_fe_op = \
                tf.train.AdamOptimizer(lr).minimize(
                        wd_param * crit.wd_loss, 
                        var_list=train.theta_G)
            
        # Regularization loss
        train.variables = train.theta_C + train.theta_D + train.theta_G
        #train.variables = tf.trainable_variables()
        train.l2_loss = \
            l2_param * tf.add_n(
                    [tf.nn.l2_loss(v) 
                    for v in train.variables if 'bias' not in v.name])
                
        # # Train FE and CL with cl loss
        # if clf_t == None:
        #     train.clf_loss = clf.loss +  train.l2_loss
        # else:
        #     train.clf_loss = clf.loss + clf_t.loss + train.l2_loss
        # # batch norm to be added here    
        # train.cl_fe_op = \
        #     tf.train.AdamOptimizer(lr).minimize(
        #             train.clf_loss, 
        #             var_list=train.theta_G + train.theta_C)
            
        # Training FE and CL with cl and wd loss
        if clf_t == None:
            train.total_loss = clf.loss +  train.l2_loss + wd_param * crit.wd_loss
        else:
            train.total_loss = clf.loss + clf_t.loss + train.l2_loss + wd_param * crit.wd_loss
        
        train.optimizer = tf.train.AdamOptimizer(lr)
        
        train.grads = train.optimizer.compute_gradients(train.total_loss, var_list=train.theta_G + train.theta_C)
        
        update_ops = tf.get_collection(tf.GraphKeys().UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train.train_op = train.optimizer.apply_gradients(train.grads, global_step=train.global_step)
                
        return train
    
    
    def train_adapter_without_critic(self, clf, l2_param, lr, clf_t=None):
        train = Training()

        train.theta_C = \
            [v for v in tf.trainable_variables() if 'classifier' in v.name]
  
        train.theta_G = \
            [v for v in tf.trainable_variables() 
            if 'feature_extractor' in v.name]
            

        
        train.variables = train.theta_C + train.theta_G
        #train.variables = tf.trainable_variables()
        train.l2_loss = \
            l2_param * tf.add_n(
                    [tf.nn.l2_loss(v) 
                    for v in train.variables if 'bias' not in v.name])
        if clf_t == None:
            train.total_loss = clf.loss +  train.l2_loss 
        else:
            train.total_loss = clf.loss + clf_t.loss + train.l2_loss 
        
        

        # Training generator and classifier
        train.optimizer = tf.train.AdamOptimizer(lr)
        
        train.grads = train.optimizer.compute_gradients(train.total_loss, var_list=train.theta_G + train.theta_C)
        
        update_ops = tf.get_collection(tf.GraphKeys().UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train.train_op = train.optimizer.apply_gradients(train.grads, global_step=train.global_step)
        return train
    
    
    
    
    def extract_features(self, features='shared'):
        if features == 'shared':
            run_ops = [self.slicer.h_s, self.slicer.h_t]
        elif features == 'separated':
            run_ops = [self.fe.output, self.fe_2.output]
        else:
            raise ValueError("features parameters must be 'shared' or 'separated'")
        return run_ops
    
   









        
