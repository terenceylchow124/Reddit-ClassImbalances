# Reddit-ClassImbalances

In this project, we apply three sampling method for handling the class imbalance problem on Reddit Joke Dataset:
- Resampling 
- Easy Data Augmentation
- Near-Miss

# Dataset Preparation 
We mainly use Reddit Joke Dataset in this project
- Go to [this official repository](https://github.com/orionw/RedditHumorDetection)
- Download the "reddit_full_data.csv" under ./full_datasets/reddit_jokes/
- We also provide the same dataset, called "reddit.txt" under ./data

# Run sampling method 
Please go to the corresponding directory and run the main python file. Feel free to modifiy few hyper-parameters related to learning alogirthm and cross-validation, i.e. EVAL_STEPS, TEST_FOLD, VAL_FOLD, INPUT_EPOCH. 
- For EDA, the python file, eda.py is borrowed from [this official repository](https://github.com/jasonwei20/eda_nlp) but with little modifaction, e.g. we skep EDA when number of words of sentence is shorter than 30. 
- For NearMiss, we use KNN classifier by default, but please feel free to modify as 'Perceptron'. 

# Hyper-parameter Result 
| Sampling      | Hyper-parameter   | Val Loss   | Val Accuracy | Val F1 | Val Precision | Val Recall |
| ------------- | ----------------- | ---------- | ------------ | ------ | ------------- | ---------- |
| Resampling    | δ = 1.00          | 0.5769     | 0.7046       | 0.7124 | 0.6981        | 0.7327     |
| Resampling    | δ = 0.75          | 0.5807     | 0.7071       | 0.7173 | 0.6948        | 0.7469     |
| Resampling    | δ = 0.50          | 0.5809     | 0.7012       | 0.7028 | 0.6968        | 0.7136     |
| NearMiss      | K = 5             | 3.3517     | 0.5343       | 0.5996 | 0.5256        | 0.6988     |
| NearMiss      | K = 7             | 6.9361     | 0.5213       | 0.6327 | 0.5133        | 0.8247     |
| NearMiss      | K = 9             | 8.9453     | 0.5207       | 0.6462 | 0.5121        | 0.8759     |
| EDA           | β = 0.05,0,0,0    | 0.8298     | 0.6744       | 0.5863 | 0.8055        | 0.4673     |
| EDA           | β = 0,0.05,0,0    | 0.8406     | 0.6750       | 0.5817 | 0.8155        | 0.4568     |
| EDA           | β = 0,0,0,0.05    | 0.7913     | 0.6806       | 0.6095 | 0.7840        | 0.5037     |
| EDA           | β = 0.03,0,0,0    | 0.8033     | 0.6840       | 0.6196 | 0.7784        | 0.5148     |
| EDA           | β = 0,0,0,0.03    | 0.7990     | 0.6787       | 0.6082 | 0.7801        | 0.5037     |
| EDA           | β = 0.03,0,0,0.03 | 0.8209     | 0.6722       | 0.5946 | 0.7799        | 0.4870     |

# Classification Result 
| Sampling                | Hyper-parameter | Train Accuracy | Val Loss   | Val Accuracy | Val F1 | Val Precision | Val Recall |
| ----------------------- | --------------- | -------------- | ---------- | ------------ | ------ | ------------- | ---------- |
| Resampling              | δ = 1.00        | 0.7707         | 0.5606     | 0.7106       | 0.7183 | 0.7030        | 0.7383     |
| NearMiss (KNN)          | K = 5           | 0.7254         | 3.0943     | 0.5543       | 0.6040 | 0.5430        | 0.6810     |
| NearMiss (Precetron)    | K = 5           | 0.7925         | 0.8188     | 0.5975       | 0.5891 | 0.6020        | 0.5773     |
| NearMiss                | β = 0,0,0,0.05  | -              | 1.5833         | 0.6790     | 0.5846       | 0.8242 | 0.4563     |
