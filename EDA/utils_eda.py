# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:21:03 2021

@author: ASUS
"""
import argparse
import pandas as pd
from tqdm import tqdm
from eda import *
import string
punctuation_list = string.punctuation

def replace_contraction(text):
    contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'can not'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'), (r'(\w+)\'ve', '\g<1> have'), 
                             (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

def replace_links(text, filler=' '):
        text = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
                      filler, text).strip()
        return text
    
def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def remove_punctuation(text):
    no_punct=[words for words in text if words not in punctuation_list]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct

def convert_utf_pronunciation(text):
    utf8_single_right = b'\xe2\x80\x99'.decode("utf8")
    utf8_single_left = b'\xe2\x80\x98'.decode("utf8")
    
    text = re.sub(utf8_single_right, "'", text)
    text = re.sub(utf8_single_left, "'", text)
    
    utf8_double_right = b'\xe2\x80\x9d'.decode("utf8")
    utf8_double_left = b'\xe2\x80\x9c'.decode("utf8")

    text = re.sub(utf8_double_right, "\"", text)
    text = re.sub(utf8_double_left, "\"", text)
    
    return text

def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    text = convert_utf_pronunciation(text)
    text = replace_contraction(text)
    text = replace_links(text, "link")
    # text = remove_numbers(text)
    # text = re.sub(r'[,”@#$%^&*)“‘(|/><";:\'\\}{]!?_',"",text)
    text = text.replace('_', ' ')
    text = remove_punctuation(text)
    text = text.replace('-', '')
    text = text.replace('\t', '')
    text = text.replace('\r', '')
    text = text.replace('\n', '')
    text = " ".join(text.split())
    return text



def get_eda_args(sr=0.1, ri=0.1, rs=0.1, rd=0.1, num_aug=6):
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha_sr", type=float, default=sr, help="percent of words in each sentence to be replaced by synonyms")
    ap.add_argument("--alpha_ri", type=float, default=ri, help="percent of words in each sentence to be inserted")
    ap.add_argument("--alpha_rs", type=float, default=rs, help="percent of words in each sentence to be swapped")
    ap.add_argument("--alpha_rd", type=float, default=rd, help="percent of words in each sentence to be deleted")
    ap.add_argument("--num_aug", type=int, default=num_aug, help="number of augmented sentences per original sentence")
    args = ap.parse_args()
    return args

def generate_aug_eda(X_train, y_train, eda_args, first):
        
    alpha_sr = eda_args.alpha_sr
    alpha_ri = eda_args.alpha_ri
    alpha_rs = eda_args.alpha_rs
    alpha_rd = eda_args.alpha_rd
    num_aug = eda_args.num_aug
    
    # create dataframe for storing augmented data
    X_train_aug = pd.DataFrame(columns = ['X'])
    y_train_aug = pd.DataFrame(columns = ['y'])
    
    # iterate the current dataset
    num_sample = len(X_train)
    for index in tqdm(range(num_sample)):
        # print(str(index), " / ", str(num_sample))
        if (first == True):
            X_train_sample = X_train.iloc[index]
            y_train_sample = y_train.iloc[index]
            if (y_train_sample):
                # perform augmentation by using eda with current hyper-parameters
                X_train_sample = cleanText(X_train_sample)
                aug_sentences = eda(X_train_sample, alpha_sr=alpha_sr, alpha_ri=alpha_ri, 
                                    alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
                
                # append every augmented data to dataset
                for aug_sentence in aug_sentences:
                    X_train_aug = X_train_aug.append({'X': aug_sentence}, ignore_index=True)
                    y_train_aug = y_train_aug.append({'y': y_train_sample}, ignore_index=True)
            else:
                 X_train_aug = X_train_aug.append({'X': X_train_sample}, ignore_index=True)
                 y_train_aug = y_train_aug.append({'y': y_train_sample}, ignore_index=True)
    
        elif (first == False):
            X_train_sample = X_train['X'].iloc[index]
            y_train_sample = y_train['y'].iloc[index]
            if (y_train_sample):
                # perform augmentation by using eda with current hyper-parameters
                X_train_sample = cleanText(X_train_sample)
                aug_sentences = eda(X_train_sample, alpha_sr=alpha_sr, alpha_ri=alpha_ri, 
                                    alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
                
                # append every augmented data to dataset
                for aug_sentence in aug_sentences:
                    X_train_aug = X_train_aug.append({'X': aug_sentence}, ignore_index=True)
                    y_train_aug = y_train_aug.append({'y': y_train_sample}, ignore_index=True)
            else:
                 X_train_aug = X_train_aug.append({'X': X_train_sample}, ignore_index=True)
                 y_train_aug = y_train_aug.append({'y': y_train_sample}, ignore_index=True)
    
            
    # print("generate augmented sentences by eda")
    return X_train_aug, y_train_aug
    