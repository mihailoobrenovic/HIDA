# -*- coding: utf-8 -*-

from sklearn.metrics import silhouette_score
import numpy as np

### Silhouette score ###

def silhouette_domains(xs_fl, xt_fl):
    n_s = xs_fl.shape[0]
    x_data = np.concatenate((xs_fl, xt_fl))
    domain_labels = np.zeros((x_data.shape[0]), dtype=np.int8)
    domain_labels[n_s:] = 1
    # Calculate silhouette score between domains
    silh_score = silhouette_score(x_data, domain_labels)
    print("Silhouette score:", silh_score)
    return silh_score

























    
