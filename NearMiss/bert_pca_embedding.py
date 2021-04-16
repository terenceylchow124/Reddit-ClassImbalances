from transformers import pipeline, AutoTokenizer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from tqdm import tqdm

def preprocessor(data, MAX_LENGTH, PCA_DIMEN):
    model_name = "bert-base-uncased"
    
    # define tokenizer for BERT model
    tokenizer = AutoTokenizer.from_pretrained(model_name)   
    encoded_seq = tokenizer.encode("i am sentence")
    
    # define feature extrator
    feature_extraction = pipeline('feature-extraction', \
                                  model=model_name, tokenizer=model_name)
    data = data.sample(frac=1).reset_index(drop=True)
    
    # example sentence to find embedding dimension, i.e. dim = 768
    features = feature_extraction("i am sentence")
    num_emb = len(features[0][1])

    # create dataframe for storing augmented data
    X_emb = pd.DataFrame(columns = ['X'])
    y_emb = pd.DataFrame(columns = ['y'])

    # iterate training data
    for index in tqdm(range(len(data))):
        
        X_sample = data['X'].iloc[index]
        y_sample = data['y'].iloc[index]
        
        # pre-truncate the data since bert accepts data with max length of 512
        if len(X_sample) > 512:
            X_sample = X_sample[:512]
            
        # convert into the embedding, size: [Nx768]
        features_list = feature_extraction(X_sample)[0]
        
        # convert list into numpy array
        temp = np.array(features_list)
        
        # truncate first 128 word embedding, size: [128x768]
        temp = temp[:MAX_LENGTH,:]
        
        # define empty array, size: [128x768]
        features = np.zeros((MAX_LENGTH, num_emb)) 
        
        # transfer word representation into unified represetation
        num_word = temp.shape[0]
        features[:num_word,:] = temp
      
        # apply pca for dimensionality reduction for each word, size: [128x50]
        pca = PCA(n_components=PCA_DIMEN)
        pca.fit(features)
        features = pca.transform(features)
        
        features = np.reshape(features, (-1,))
        # the shape is now in 1D: 6400 for every data
        X_emb = X_emb.append({'X': features}, ignore_index=True)
        y_emb = y_emb.append({'y': y_sample}, ignore_index=True)

    return pd.concat([y_emb, X_emb], axis=1)

if __name__=="__main__":  
    MAX_LENGTH = 128
    PCA_DIMEN = 50
    path = '../data/sample_val_under.tsv'
    data = pd.read_csv(path, sep='\t', encoding="utf-8", names=["y", "X"])
    emb = preprocessor(data, MAX_LENGTH, PCA_DIMEN)
    

