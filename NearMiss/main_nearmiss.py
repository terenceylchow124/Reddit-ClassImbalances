# -*- coding: utf-8 -*-
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import numpy as np
import pandas as pd
import torch
import json
import time
from bert_pca_embedding import preprocessor
from utils_nearmiss import nearmiss
from classifier import model

RANDOM_STATE = 42
TEST_FOLD = 5
VAL_FOLD = 5
MAX_LENGTH = 128
PCA_DIMEN = 50
KNeighbors = 15
K = [5,7,9]

def to_array_list(data):
    X = np.array(data['X'].tolist())
    y = np.array(data['y'].tolist())
    X = X.tolist()
    df = pd.DataFrame({'y': y, 'X': X})
    return df
    
# oversampling (OS) the majority class to given number 
def oversampling(data, num_over):
    majority = data[data['y']==0]
    minority = data[data['y']==1]
    minority_oversampled = resample(minority, 
                                     replace=True,     # sample with replacement
                                     n_samples= minority.shape[0]+num_over,    # to match majority class
                                     random_state=42) # reproducible results
    
    oversampled_data = pd.concat([majority, minority_oversampled])
    return oversampled_data

# undersampling (US) the minority class to given number 
def undersampling(data, num_under):
    majority = data[data['y']==0]
    minority = data[data['y']==1]
    majority_undersampled = resample(majority, 
                                     replace=False, # sample without replacement
                                     n_samples = majority.shape[0]-num_under,  # to match majority class
                                     random_state=RANDOM_STATE) # reproducible results
    undersampled_data = pd.concat([minority, majority_undersampled])
    
    return undersampled_data

# Inner-Cross-Validation on train and validation dataset
def train_val(X_trainval, y_trainval, temp_dir, current_K, test_count, hyper_count):
    
    # constructing the validataion set by stratified cross-validataion
    skf_train_val = StratifiedKFold(n_splits=VAL_FOLD, 
                                random_state=RANDOM_STATE, 
                                shuffle=True)
    
    skf_train_val.get_n_splits(X_trainval, y_trainval)
    fold_count = 1
    
    all_train_acc = []
    all_train_loss = []
    all_val_loss = []
    
    all_val_acc = []
    all_val_f1 = []
    all_val_precision = []
    all_val_recall = []
    
    all_num_train = []
    all_num_val = []
    all_num_under_val = []

    for train_index, val_index in skf_train_val.split(X_trainval, y_trainval):    
        
        X_train = X_trainval.iloc[train_index]
        y_train = y_trainval.iloc[train_index]
        
        X_val   = X_trainval.iloc[val_index]
        y_val  = y_trainval.iloc[val_index]
                 
        # undersampling val set by resampling
        sample_val = pd.concat([y_val, X_val], axis=1)
        num_class0, num_class1 = y_val.value_counts()
        all_num_val.append([num_class0, num_class1])
        
        # calculating the required US samples
        num_us_instance = num_class0 - num_class1
        under_sample_val = undersampling(sample_val, num_us_instance)
        num_class0, num_class1 = under_sample_val['y'].value_counts()
        all_num_under_val.append([num_class0, num_class1])   
        
        # pre-sampling undersampled val data by pretrained BERT with PCA
        print("-"*70)
        print("START PROCESSING VAL SET")
        under_sample_val_emb = preprocessor(under_sample_val, MAX_LENGTH, PCA_DIMEN)
        under_sample_val_emb = to_array_list(under_sample_val_emb)
        
        # saving undersampled val data
        under_val_path = temp_dir + '/sample_val_under_emb.tsv'
        under_sample_val_emb.to_csv(under_val_path, sep='\t',encoding="utf-8", index=False, header=False)
        
        del X_val, y_val
        del sample_val
        del under_sample_val
        del under_sample_val_emb
    
        # preprocessing train data
        print("-"*70)
        print("START PROCESSING TRAIN SET")
        sample_train = pd.concat([y_train, X_train], axis=1)
        
        # pre-sampling train data by pretrained BERT with PCA
        sample_train_emb = preprocessor(sample_train, MAX_LENGTH, PCA_DIMEN)
    
        # performing undersampling by NearMiss
        print("-"*70)
        print("START  SAMPLING TRAIN SET [{}] ".format(fold_count))
        start_time = time.time()
        sample_train_nearmiss = nearmiss(sample_train_emb, current_K)
        end_time = time.time()
        print("FINISH SAMPLING TRAIN SET [{}]: {}".format(fold_count, 
                                                      (end_time - start_time)))
        
        # saving undersampled train data
        train_path = temp_dir + '/sample_train_nearmiss.tsv'
        sample_train_nearmiss.to_csv(train_path, sep='\t',encoding="utf-8", index=False, header=False)
        
        num_class0, num_class1 = sample_train_nearmiss['y'].value_counts()
        all_num_train.append([num_class0, num_class1])
                      
        del  X_train, y_train, sample_train
        
        # processing to model classifier (preceptron or KNN)
        history = model(train_path, under_val_path, test_count, \
                        hyper_count, fold_count, model='KNN', test=False)
        
        # calculating average train loss and accuracy
        all_train_loss.append(history['train_loss'])
        all_train_acc.append(history['train_acc'])
        
        # calculating average val loss, accuracy, F1, precision and recall
        all_val_loss.append(history['val_loss'])
        all_val_acc.append(history['val_acc'])
        all_val_f1.append(history['val_f1'])
        all_val_precision.append(history['val_precision'])
        all_val_recall.append(history['val_recall'])
        
        # increamenting the fold index, and repeat above process        
        fold_count = fold_count + 1
    
    # return back the ICV log
    results = {'final_all_train_acc': all_train_acc,
               'final_all_train_loss': all_train_loss,
               'final_all_val_loss': all_val_loss,
               'final_all_val_acc': all_val_acc,
               'final_all_val_f1': all_val_f1,
               'final_all_val_precision': all_val_precision,
               'final_all_val_recall': all_val_recall,
               'final_avg_val_loss': np.mean(all_val_loss),
               'train_distribution': all_num_train,
               'val_distribution': all_num_val,
               'under val val_distribution': all_num_under_val
               }
    return results

def main(X_trainval, y_trainval, temp_dir, test_count):
    
    hyper_count = 1
    
    hyper_result = []
    best_hyper_val_loss = 100
    best_hyper = hyper_count
    
    # iterating over several hyper-parameter setting, K=5,7,9
    for current_K in K:
        print("-"*70)
        print("HYPER-PARAMETERS TUNING, CURRENTLY USING: ", str(current_K))
             
        # creating train and val set by splitting and augmenting
        # performing ICV to otain the best hyper-parameter setting 
        result = train_val(X_trainval, y_trainval, 
                           temp_dir, current_K, test_count, hyper_count)
        hyper_result.append(result)
                           
        # recording best hyper index
        if result['final_avg_val_loss'] < best_hyper_val_loss:
            best_hyper_val_loss = result['final_avg_val_loss']                   
            best_hyper = hyper_count  
            
        hyper_count = hyper_count + 1 
        
    return best_hyper_val_loss, best_hyper, hyper_result

if __name__=="__main__":
    
    # constructing the dataset
    path = '../data/reddit.txt'
    temp_dir = '../data'
    
    reddit = pd.read_csv(path, names=['y','X'], sep='\t',encoding="utf-8")
    X = reddit['X']
    y = reddit['y']
    
    best_hyper = 1
    best_fold = 0
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    test_f1 = []
    test_precision = []
    test_recall = []
    
    all_num_trainval = []
    all_num_test = []
    all_num_under_test = []
    
    # constructing stratified K-Folds for Outer-Cross-Validation (OCV)
    # splitting into trainval & test datasets 
    skf = StratifiedKFold(n_splits=TEST_FOLD, random_state=RANDOM_STATE, 
                          shuffle=True)
    
    skf.get_n_splits(X, y)
    test_count = 1
    for trainval_index, test_index in skf.split(X, y):
        X_trainval, X_test = X[trainval_index], X[test_index]
        y_trainval, y_test = y[trainval_index], y[test_index]

        # undersampling test set
        num_class0, num_class1 = y_test.value_counts()
        all_num_test.append([num_class0, num_class1])
        num_us_instance = num_class0 - num_class1
        sample_test = pd.concat([y_test, X_test], axis=1)
        under_sample_test = undersampling(sample_test, num_us_instance)
         
        num_class0, num_class1 = under_sample_test['y'].value_counts()
        all_num_under_test.append([num_class0, num_class1])
        
        # pre-sampling undersampled test data by pretrained BERT with PCA
        print("-"*70)
        print("START PROCESSING TEST SET")
        under_sample_test_emb = preprocessor(under_sample_test, MAX_LENGTH, PCA_DIMEN)
        under_sample_test_emb = to_array_list(under_sample_test_emb)
        
        # saving undersampled test data  
        under_test_path = temp_dir + '/sample_test_under_emb.tsv'
        under_sample_test_emb.to_csv(under_test_path, sep='\t',encoding="utf-8", index=False, header=False)
             
        del X_test, y_test, under_sample_test, sample_test, 
        del under_sample_test_emb
            
        
        if test_count == 1:
            # process to main function with current train and val path
            # return the best number of fold and corresponding hyper setting
            best_hyper_val_loss, best_hyper, \
                hyper_result =  main(X_trainval, y_trainval, temp_dir, test_count)
            
        # pre-sampling undersampled trainval data by pretrained BERT with PCA
        print("-"*70)
        print("START PROCESSING TRAINVAL SET")
        sample_trainval = pd.concat([y_trainval, X_trainval], axis=1)
        sample_trainval_emb = preprocessor(sample_trainval, MAX_LENGTH, PCA_DIMEN)
        
        del  X_trainval, y_trainval, sample_trainval
        current_K = K[best_hyper-1]
        
        # performing undersampling by NearMiss
        print("-"*70)
        print("START  SAMPLING TRAINVAL SET [{}] ".format(test_count))
        start_time = time.time()
        sample_trainval_nearmiss = nearmiss(sample_trainval_emb, current_K)
        end_time = time.time()
        print("FINISH SAMPLING TRAINVAL SET [{}]: {}".format(test_count, 
                                                      (end_time - start_time)))
        
        # saving undersampled trainval data
        trainval_path = temp_dir + '/sample_trainval_nearmiss.tsv'
        sample_trainval_nearmiss.to_csv(trainval_path, sep='\t',encoding="utf-8", index=False, header=False)
        
        num_class0, num_class1 = sample_trainval_nearmiss['y'].value_counts()
        all_num_trainval.append([num_class0, num_class1])
        
        # processing to model classifier (preceptron or KNN)
        history = model(trainval_path, under_test_path, test_count, \
                        best_hyper, best_fold, model='Perceptron', test=True)
             
      
        # calculating average trainval loss and accuracy
        train_loss.append(history['train_loss'])
        train_acc.append(history['train_acc'])

        
        # calculating average test loss accuracy, F1, precision and recall
        test_loss.append(history['val_loss'])
        test_acc.append(history['val_acc'])
        test_f1.append(history['val_f1'])
        test_precision.append(history['val_precision'])
        test_recall.append(history['val_recall'])

        test_count = test_count + 1
    
    # creating the resulted dict for saving as json file
    trainval_test_result = {'train_acc': train_acc,
                            'train_loss': train_loss,
                            'test_loss': test_loss, 
                            'test_acc': test_acc,
                            'test_f1': test_f1, 
                            'test_precision': test_precision,
                            'test_recall': test_recall,
                            'avg_train_loss': np.mean(train_loss),
                            'avg_train_acc': np.mean(train_acc),
                            'avg_test_loss': np.mean(test_loss),
                            'avg_test_acc': np.mean(test_acc),
                            'avg_test_f1': np.mean(test_f1),
                            'avg_test_precision': np.mean(test_precision),
                            'avg_test_recall': np.mean(test_recall),
                            'best_hyper_setting': best_hyper,
                            'hyper_cv_log': hyper_result,
                            'trainval distribution': all_num_trainval,
                            'test distribution': all_num_test,
                            'under test distribution': all_num_under_test
                            }
    
    with open(f'./result/trainval_test_result.json', 'w') as output_file:
        json.dump(trainval_test_result, output_file)
    
    print("average train loss: ", np.mean(train_loss))
    print("average test loss: ", np.mean(test_loss))
    print("average test accuracy: ", np.mean(test_acc))
    print("average test f1: ", np.mean(test_f1))
    print("average test precision: ", np.mean(test_precision))
    print("average test recall: ", np.mean(test_recall))