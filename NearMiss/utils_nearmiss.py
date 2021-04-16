# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 23:01:57 2021

@author: ASUS
"""
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
from collections import Counter
from matplotlib import pyplot
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import pandas as pd
pyplot.xlim([-3, 4])
pyplot.ylim([-0.6, 3])

# plotting using tsne: also works but takes time
# haven't used in this project    
def plot_tsne(data_X, y, use, tsne_df_ori=None):
    
    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj= tsne.fit_transform(data_X)
        
    tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                            'Y':tsne_obj[:,1],
                            'digit':y})
    
    if not use:
        sns.scatterplot(x="X", y="Y", 
                        hue="digit",
                        palette=['blue','red'],
                        legend='full',
                        data=tsne_df);
    else: 
        sns.scatterplot(x="X", y="Y", 
                hue="digit",
                palette=['blue','red'],
                legend='full',
                data=tsne_df_ori);
        sns.scatterplot(x="X", y="Y", 
                hue="digit",
                palette=['purple','green'],
                legend='full',
                data=tsne_df);
    pyplot.show()
    return tsne_df

def plot_scatter(X, y, counter, max_X, min_X, max_y, min_y):
    # scatter plot of examples by class label
    for label, _ in counter.items():
     	row_ix = np.where(y == label)[0]
     	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    pyplot.legend()
    
    # set the x and y-limit to have same plotting window 
    pyplot.xlim([min_X, max_X])
    pyplot.ylim([min_y, max_y])
    pyplot.legend(loc = "upper right")
    pyplot.show()
    
def nearmiss(data, k):
    
    X = np.array(data['X'].tolist())
    y = np.array(data['y'].tolist())
    
    max_X = np.max(X[:,0])
    min_X = np.min(X[:,0])
    
    max_y = np.max(X[:,1])
    min_y = np.min(X[:,1])
    
    counter = Counter(y)
    
    print("Before undersampling:", counter)
    plot_scatter(X, y, counter, max_X, min_X, max_y, min_y)

    # define the undersampling method, version: 3, k is a target hyper-parameter
    undersample = NearMiss(version=3, n_neighbors_ver3=k)
    
    # perform NearMiss sampling
    X_sampled, y_sampled = undersample.fit_resample(X, y)
    X_list = X_sampled.tolist()
    sample_data = pd.DataFrame({'y': y_sampled, 'X': X_list})
    counter = Counter(y_sampled)
    print("After undersampling:", counter)
    
    plot_scatter(X_sampled, y_sampled, counter, max_X, min_X, max_y, min_y)
    
    return sample_data

if __name__=="__main__": 
    data = ""
    NEARMISS_K = 3
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
 	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
    sampled_data = nearmiss(data, NEARMISS_K)