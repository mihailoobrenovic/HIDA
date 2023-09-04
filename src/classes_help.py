#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import warnings







######################select classes##############

rgb_resisc_classes = ['crop', 'forest', 'industrial', 'residential', 'river']
ms_eurosat_classes = ['crop', 'forest', 'industrial', 'residential', 'river', 
                      'herbaceousvegetation', 'highway', 'pasture', 'permanentcrop', 
                      'sealake']




def selectClasses(mode="random", source=True, nbr_classes = 5, class_idx_list = None, dataset="space"):
    """
    To select classes in the flow from directory
    
    Parameters
    ----------
    mode : String, "random" or anything else, optional
        DESCRIPTION. The default is "random" : selects randoms classes
        
    source : Boolean, optional
        DESCRIPTION. The default is True. Source or Target domain
        
    nbr_classes : int, optional
        DESCRIPTION. The default is 5.
        
    class_idx_list : list of integers, optional
        DESCRIPTION. The default is None. Should be passed if mode ! "random"
        
    dataset : string, "space" or "digit" or "medical"

    Returns
    -------
    a sorted list of class names
        eg. ['crop', 'forest', 'industrial]

    """
    
    
    if dataset == "space":
        if source:
            domain_class = rgb_resisc_classes
        else:
            domain_class = ms_eurosat_classes
            
            
    max_class = len(domain_class)
   
        
        
    if nbr_classes > max_class :
        warnings.warn("nbr_classes should be < " + str(max_class))
        nbr_classes = max_class
    
    
    
    classes_list =[]
    
    if mode == "random":
        rd_idx = np.arange(max_class)
       
        np.random.shuffle(rd_idx)
       
        for i in range(nbr_classes):
            idx = rd_idx[i]
            
            classes_list.append(domain_class[idx])
            
            
    elif class_idx_list != None:
        class_idx_list = class_idx_list[:nbr_classes]
        classes_list = [domain_class[idx] for idx in  class_idx_list]
        
    classes_list.sort()
    return classes_list


