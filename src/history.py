# -*- coding: utf-8 -*-
import pickle
import numpy as np

class History():
    
    def __init__(self, path='pickles/wdgrl.pkl', steps_int=1):
        self.reset()
        self.path = path
        self.steps_interval = steps_int
        self.steps_per_epoch = None
        
        
    def reset(self):
        self.step_losses = []
        
        self.t_losses = []
        self.v_losses = []
        
        self.min_model_loss = np.inf
        self.max_model_loss = -np.inf
        
        self.checkpoint = False
        
    def set_steps_per_epoch(self, epoch_iters):
        self.steps_per_epoch = epoch_iters
        
        
   
    def append_step(self, step_losses):
        self.step_losses.append(step_losses)
            

    def append_epoch(self, t_losses, v_losses, min_or_max='min'):
        self.checkpoint = False
        model_loss = v_losses['model_loss']
        if min_or_max == 'min':
            if model_loss < self.min_model_loss:
                self.min_model_loss = model_loss
                self.checkpoint = True
        elif min_or_max == 'max':
            if model_loss > self.max_model_loss:
                self.max_model_loss = model_loss
                self.checkpoint = True
        self.t_losses.append(t_losses)
        self.v_losses.append(v_losses)
        
        
    def save_epoch(self):
        pickle.dump([self.step_losses, self.t_losses, self.v_losses, self.steps_per_epoch], 
                    open(self.path,'wb'))
        
        
    def load_epoch(self):
        [self.step_losses, self.t_losses, self.v_losses, self.steps_per_epoch] = \
            pickle.load(open(self.path,'rb'))

        
    def plot_basic(self, values, labels, scale="linear", epoch=False, plot_path=None):
        from matplotlib import pyplot as plt
        
        n = len(values[0])
        x = np.arange(1,n+1)
        if not epoch:
            x *= self.steps_interval
        fig = plt.figure()
        for v, l in zip(values, labels):
            plt.plot(x, v, label=l)
        plt.yscale(scale)
        plt.legend()
        if plot_path == None:
            plt.show()
        else:
            fig.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches = 0)
        
        
    def plot_basic_subplots(self, values_mat, labels_mat, rows, cols, scale, 
                            epoch=False, plot_path=None):
        from matplotlib import pyplot as plt
        fig_width_ratio = 20/3
        fig_height_ratio = 4.25
        fig_width = cols * fig_width_ratio
        fig_height = rows * fig_height_ratio
        fig, axs = plt.subplots(rows, cols, figsize=(fig_width,fig_height), 
                                squeeze=False)
        for i in range(rows):
            for j in range(cols):
                values = values_mat[i][j]
                labels = labels_mat[i][j]
                if values != []:
                    n = len(values[0])
                    x = np.arange(1,n+1)
                    if not epoch:
                        x *= self.steps_interval
                    for v, l in zip(values, labels):
                        axs[i][j].plot(x, v, label=l)
                    axs[i][j].set_yscale(scale[i][j])
                    axs[i][j].legend()
        if plot_path == None:
            plt.show()
        else:
            fig.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches = 0)

        
        