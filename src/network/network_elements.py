# -*- coding: utf-8 -*-

class Input():
    def __init__(self, inp_shape, num_channels, num_class):
        self.inp_shape = inp_shape
        self.num_channels = num_channels
        self.num_class = num_class
        self.X = None
        self.y_true = None
        self.train_flag = None
        self.y_true_one_hot = None
   
     
class FeatureExtractor():
    def __init__(self):
        self.conv_layers = []
        self.W = []
        self.b = []
        self.fc_layers = []
        self.add_bs = []
        self.multipls = []
        self.flattened = None
        self.output = None
    
    
class Slicer():
    def __init__(self):
        self.h_s = None
        self.h_t = None
        self.ys_true = None
        self.yt_true = None

        
        
# Used for WDGRL with 2 generators
# If training - take the output from the first generator
# If testing - take the output from the second generator        
class Condition():
    def __init__(self):
        self.h = None
        self.y_one_hot = None
        self.y_label = None
        
        
class Merger():
    def __init__(self):
        self.h = None
        self.y_one_hot = None
        self.y_label = None
       
        
class Classifier():
    def __init__(self):
        self.fc_layers = []
        self.add_bs = []
        self.multipls = []
        self.W = None
        self.b = None
        self.pred_logit = None
        self.pred_softmax = None
        self.y_pred = None
        self.loss = None
        self.total_loss = None
        self.conf_mat = None
       
        
class Critic():
    def __init__(self):
        self.W = []
        self.b = []
        self.alpha = None
        self.differences = None
        self.interpolates = None
        self.h_whole = None
        self.fc_layers = []
        self.output = None
        self.out_s = None
        self.out_t = None
        self.wd_loss = None
        self.out_s_sum = None
        self.out_t_sum = None
        

        
          
class Tensorboard():
    def __init__(self):
        self.merged = None
        self.writer = None



