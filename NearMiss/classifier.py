# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 01:55:48 2021

@author: ASUS
"""
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    f1_score, confusion_matrix, classification_report
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
KNN_K = 15
class_names = ['Not-Funny', 'Funny']

# preprocess dataframe to array 
def df_to_array(X):
    X_array  = []
    for ind in tqdm(range(len(X))):
        data = X[ind]
        data = data[1:-2]
        datalist = data.split(',')
        result = np.array([float(i) for i in datalist])
        X_array.append(result)
    return np.array(X_array)
    
# calculate the result metrics by using sklearn packages
def compute_metrics(labels, preds, probs):
    loss = log_loss(y_true=labels, y_pred=probs)
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    return  loss, accuracy, f1, precision, recall

def print_val_result(history, loss, accuracy, f1, precision, recall):
    print("eval loss     :", loss)
    print("eval accuracy :", accuracy)
    print("eval f1       :", f1)
    print("eval precision:", precision)
    print("eval recall   :", recall)

    history['val_loss'] = loss
    history['val_acc'] = accuracy
    history['val_f1'] = f1
    history['val_precision'] = precision
    history['val_recall'] =  recall
    return history

# KNN classifier
def KNNModel(X, y):
    classifier = KNeighborsClassifier(n_neighbors=KNN_K)
    classifier.fit(X, y)
    return classifier

# perceptron classifier
def PerceptronModel(X, y):
    pmodel = Perceptron(tol=1e-3, random_state=0)
    classifier = CalibratedClassifierCV(pmodel, cv=3, method='isotonic')
    classifier.fit(X, y)
    return classifier

# main function
def model(train_path, val_path, test_count, hyper_count, fold_count, \
          model='KNN', test=False):
    
    train_data = pd.read_csv(train_path, sep='\t', encoding="utf-8", names=["y", "X"])
    val_data = pd.read_csv(val_path, sep='\t', encoding="utf-8", names=["y", "X"])
    
    X_train = np.array(train_data['X'].tolist())
    y_train = np.array(train_data['y'].tolist())
    
    X_val = np.array(val_data['X'].tolist())
    y_val = np.array(val_data['y'].tolist())
    
    # convert the string list from dataframe to numpy array
    print("-"*70)
    print("START PREPARE DATA FOR CLASSIFIER")
    X_train = df_to_array(X_train)
    X_val = df_to_array(X_val)
    print("FINISH PREPARE DATA FOR CLASSIFIER")
    
    # data preprocessing 
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_val = min_max_scaler.transform(X_val)
    
    # train process
    print("-"*70)
    print("START TRAINING")
    
    if model == 'KNN':
        classifier = KNNModel(X_train, y_train)
    elif model == 'Perceptron':
        classifier = PerceptronModel(X_train, y_train)
        
    y_train_prob = classifier.predict_proba(X_train)
    y_train_pred = classifier.predict(X_train)
    
    history = {}
    loss, accuracy, _, _, _ = compute_metrics(y_train, y_train_pred, y_train_prob)
    
    print("train loss    : ", loss)
    print("train accuracy: ", accuracy)
    
    history['train_loss'] = loss
    history['train_acc'] = accuracy
    
    # validate/test process
    print("-"*70)
    print("START VALIDATING")
    
    y_val_prob = classifier.predict_proba(X_val)
    y_val_pred = classifier.predict(X_val)
    
    loss, accuracy, f1, precision, recall = compute_metrics(y_val, y_val_pred, y_val_prob)
    
    history = print_val_result(history, loss, accuracy, f1, precision, recall)

    # save log result
    if test==False:
        with open(f'../result/final_trainer_state_{test_count}_{hyper_count}_{fold_count}.json', 'w') as output_file:
                json.dump(history, output_file)
    else:
        with open(f'../result/final_test_state_{test_count}.json', 'w') as output_file:
            json.dump(history, output_file)
            
    if test == True:
        report = classification_report(y_val, y_val_pred, target_names=class_names, digits=4)
        report_path = "../result/report_{}.txt".format(test_count) 
        text_file = open(report_path, "w")
        text_file.write(report)
            
    return history

if __name__=="__main__": 
    temp_dir = '../data'
    train_path = temp_dir + '/sample_train_nearmiss.tsv'
    val_path = temp_dir + '/sample_val_under_emb.tsv'
    history = model(train_path, val_path, 1,1,1, model='KNN')
    
    
    
    

