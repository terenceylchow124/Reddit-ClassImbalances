# -*- coding: utf-8 -*-

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import time
import torch
import numpy as np
import pandas as pd
import json
from utils_eda import *
from bert_April6 import bert
RANDOM_STATE = 42
EVAL_STEPS = 1200
TEST_FOLD = 5
VAL_FOLD = 5
INPUT_EPOCH = 3
alpha = torch.tensor([[0.05, 0, 0, 0], [0, 0.05, 0, 0], [0, 0, 0, 0.05], 
[0.03, 0, 0, 0], [0, 0, 0, 0.03], [0.03, 0, 0, 0.03]])

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
def undersampling(data):
    majority = data[data['y']==0]
    minority = data[data['y']==1]
    majority_undersampled = resample(majority, 
                                     replace=False, # sample without replacement
                                     n_samples = minority.shape[0],  # to match majority class
                                     random_state=RANDOM_STATE) # reproducible results
    
    undersampled_data = pd.concat([minority, majority_undersampled])
    
    return undersampled_data

# Inner-Cross-Validation on train and validation dataset
def train_val(X_trainval, y_trainval, temp_dir, current_alpha, test_count, hyper_count):
    
    # construct the validataion set by stratified cross-validataion
    skf_train_val = StratifiedKFold(n_splits=VAL_FOLD, 
                                random_state=RANDOM_STATE, 
                                shuffle=True)
    
    skf_train_val.get_n_splits(X_trainval, y_trainval)
    fold_count = 1
    
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
        
        # upsampling the minority class for rearching the semi-balanced dataset
        start_time = time.time()
        print("-"*70)
        print("START  SAMPLING TRAIN SET [{}] ".format(fold_count))    
        num_class0, num_class1 = y_train.value_counts()
        diff = num_class0 - num_class1
        num_os_instance = int(diff*0.5)
        temp = pd.concat([y_train, X_train], axis=1)
        sample_train = oversampling(temp, num_os_instance)
        
        # presampling
        eda_args = get_eda_args(current_alpha[0], current_alpha[1], 
                                current_alpha[2], current_alpha[3], num_aug=0)
        X_train_aug, y_train_aug =  generate_aug_eda(sample_train['X'], 
                                                     sample_train['y'],
                                                     eda_args, first=True)
        
        # calculating the required OS samples
        num_class0, num_class1 = y_train_aug.value_counts()
        aug_factor = int(num_class0/num_class1)
        
        # performing EDA for training dataset only 
        eda_args = get_eda_args(current_alpha[0], current_alpha[1], 
                        current_alpha[2], current_alpha[3], num_aug=aug_factor)
        X_train_aug, y_train_aug =  generate_aug_eda(X_train_aug, y_train_aug,
                                               eda_args, first=False)
                
        end_time = time.time()
        print("FINISH SAMPLING TRAIN SET [{}]: {}".format(fold_count,
                                                         (end_time - start_time)))
        
                
        # undersampling val set by resampling
        sample_val = pd.concat([y_val, X_val], axis=1)
        under_sample_val = undersampling(sample_val)
        
        # saving undersampled val data
        under_val_path = temp_dir + '/sample_val_under.tsv'
        under_sample_val.to_csv(under_val_path, sep='\t',encoding="utf-8", index=False, header=False)
        
        num_class0, num_class1 = under_sample_val['y'].value_counts()
        all_num_under_val.append([num_class0, num_class1])
        
        num_class0, num_class1 = y_val.value_counts()
        all_num_val.append([num_class0, num_class1])
        
        #  saving sampled train data
        sample_train = pd.concat([y_train_aug, X_train_aug], axis=1)
        train_path = temp_dir + '/sample_train_aug.tsv'
        sample_train.to_csv(train_path, sep='\t',encoding="utf-8", index=False, header=False)
        
        num_class0, num_class1 = y_train_aug.value_counts()
        all_num_train.append([num_class0, num_class1])
        
        del sample_train, X_train, y_train
        del sample_val, X_val, y_val
        del X_train_aug, y_train_aug
        
        # processing to model classifier (BERT)
        history = bert(train_path, under_val_path, INPUT_EPOCH, EVAL_STEPS,
                       test_count, hyper_count, fold_count, predict=False)
        
        
        # calculating average train loss
        all_train_loss.append(history['train_loss'])
               
        # calculating average val loss, accuracy, f1, precision and recall
        all_val_loss.append(history['val_loss'])
        all_val_acc.append(history['val_acc'])
        all_val_f1.append(history['val_f1'])
        all_val_precision.append(history['val_precision'])
        all_val_recall.append(history['val_recall'])
             
        # increamenting the fold index, and repeat above process        
        fold_count = fold_count + 1
        
    results = {'final_all_train_loss': all_train_loss,
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
    
    for current_alpha in alpha:
        print("-"*70)
        print("HYPER-PARAMETERS TUNING, CURRENTLY USING: ", str(current_ration))
        
        # creating train and val set by splitting and augmenting
        # performing ICV to otain the best hyper-parameter setting 
        result = train_val(X_trainval, y_trainval, 
                           temp_dir, current_alpha, test_count, hyper_count)
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
    
    best_hyper = 3
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
        print("-"*70)
        print("START PROCESSING TEST SET")
        sample_test = pd.concat([y_test, X_test], axis=1)
        sample_test = undersampling(sample_test)
        
        # presampling test data
        current_alpha = alpha[best_hyper-1]
        eda_args = get_eda_args(current_alpha[0], current_alpha[1], 
                                current_alpha[2], current_alpha[3], num_aug=0)
        X_under_sample_test, y_under_sample_test =  generate_aug_eda(sample_test['X'], 
                                                                     sample_test['y'],
                                                                     eda_args, first=True)
        under_sample_test = pd.concat([y_under_sample_test, X_under_sample_test], axis=1)  
        sample_test = sample_test.sample(frac=1).reset_index(drop=True)
        under_sample_test = under_sample_test.sample(frac=1).reset_index(drop=True)   
        
        # saving both original and undersampled test data 
        test_path = temp_dir + '/sample_test.tsv'
        sample_test.to_csv(test_path, sep='\t',encoding="utf-8", index=False, header=False)
        
        under_test_path = temp_dir + '/sample_test_under.tsv'
        under_sample_test.to_csv(under_test_path, sep='\t',encoding="utf-8", index=False, header=False)
       
        num_class0, num_class1 = under_sample_test['y'].value_counts()
        all_num_under_test.append([num_class0, num_class1])
        
        num_class0, num_class1 = y_test.value_counts()
        all_num_test.append([num_class0, num_class1])
        
        del X_test, y_test, sample_test

        if test_count == 1:
            # process to main function with current train and val path
            # return the best number of fold and corresponding hyper setting
            best_hyper_val_loss, best_hyper, \
                hyper_result =  main(X_trainval, y_trainval, temp_dir, test_count)
           
        # upsampling the minority class for rearching the semi-balanced datasetprint("-"*70)
        print("START  SAMPLING TRAINVAL SET [{}] ".format(test_count))
        start_time = time.time()
        
        current_alpha = alpha[best_hyper-1]
        num_class0, num_class1 = y_trainval.value_counts()
        diff = num_class0 - num_class1
        num_os_instance = int(diff*0.5)
        temp = pd.concat([y_trainval, X_trainval], axis=1)
        sample_trainval = oversampling(temp, num_os_instance)

        # presampling
        eda_args = get_eda_args(current_alpha[0], current_alpha[1], 
                                current_alpha[2], current_alpha[3], num_aug=0)
        X_trainval_aug, y_trainval_aug =  generate_aug_eda(sample_trainval['X'], 
                                                           sample_trainval['y'],
                                                           eda_args, first=True)
                                                     
        # calculating the required OS samples
        num_class0, num_class1 = y_trainval_aug.value_counts()
        aug_factor = int(num_class0/num_class1)
        
        # performing EDA for training dataset only 
        eda_args = get_eda_args(current_alpha[0], current_alpha[1], 
                                current_alpha[2], current_alpha[3], num_aug=aug_factor)
        X_trainval_aug, y_trainval_aug =  generate_aug_eda(X_trainval_aug, 
                                                           y_trainval_aug,
                                                           eda_args, first=False)
                                               
        end_time = time.time()
        print("FINISH SAMPLING TRAIN SET [{}]: {}".format(fold_count,
                                                         (end_time - start_time)))
                                               
        #  saving sampled train data
        sample_trainval = pd.concat([y_trainval_aug, X_trainval_aug], axis=1)
        
        sample_trainval = sample_trainval.sample(frac=1).reset_index(drop=True)
        
        trainval_path = temp_dir + '/sample_trainval_aug.tsv'
        sample_trainval.to_csv(trainval_path, sep='\t',encoding="utf-8", index=False, header=False)
        num_class0, num_class1 = y_trainval_aug.value_counts()
        all_num_trainval.append([num_class0, num_class1])
        
        del X_trainval, y_trainval
        del sample_trainval, X_trainval_aug, y_trainval_aug
        
        # processing to model classifier (BERT)
        history = bert(trainval_path, under_test_path, INPUT_EPOCH, EVAL_STEPS,
                       test_count, best_hyper, best_fold, predict=True)
        
        # calculating average trainval loss and accuracy
        train_loss.append(history['train_loss'])
        train_acc.append(history['train_acc'])
        
        # calculating average test loss, accuracy, f1, precision and recall
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