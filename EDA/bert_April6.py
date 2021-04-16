import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    f1_score, confusion_matrix, classification_report
import torch
import json
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class_names = ['Not-Funny', 'Funny']

# define trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# copy folder
def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

# create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def bert(train_path, val_path, INPUT_EPOCH, EVAL_STEPS, test_count, \
         hyper_count, fold_count, predict):

    # define pretrained tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # read data
    train_data = pd.read_csv(train_path, sep='\t', encoding="utf-8", names=["y", "X"])
    val_data = pd.read_csv(val_path, sep='\t', encoding="utf-8", names=["y", "X"])

    # preprocess data
    X_train = list(train_data["X"])
    y_train = list(train_data["y"])

    X_val = list(val_data["X"])
    y_val = list(val_data["y"])

    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=128)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=128)

    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    # define trainer
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=INPUT_EPOCH,
        save_steps=3000,
        seed=0,
        load_best_model_at_end=True,
    )

    # train pre-trained BERT model
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()

    # find the final stored model
    result = os.listdir("./output")[-1]

    with open('./output/{}/trainer_state.json'.format(result)) as f:
        data = json.load(f)

        # not predict (test) mode
        if predict==False:
            with open(f'./result/final_trainer_state_{test_count}_{hyper_count}_{fold_count}_.json', 'w') as output_file:
                json.dump(data, output_file)
        else:
            with open(f'./result/final_test_state_{test_count}.json', 'w') as output_file:
                json.dump(data, output_file)

    # retrieve best training loss, eval loss and accuracy
    best = data['best_model_checkpoint'].split("-")[-1]
    history = {}
    history['train_acc'] = 0
    history['train_loss'] = 0
    print(data)
    print(data['log_history'])
    for i in data['log_history']:
        print(i)
        if i['step'] == int(best):
            if 'loss' in i:
                print("training loss:\t", i['loss'])
                history['train_loss'] = i['loss']
            if 'eval_accuracy' in i:
                print("eval loss:\t", i['eval_loss'])
                print("eval accuracy:\t", i['eval_accuracy'])
                print("eval f1:\t", i['eval_f1'])
                print("eval precision:\t", i['eval_precision'])
                print("eval recall:\t", i['eval_recall'])

                history['val_loss'] = i['eval_loss']
                history['val_acc'] = i['eval_accuracy']
                history['val_f1'] = i['eval_f1']
                history['val_precision'] = i['eval_precision']
                history['val_recall'] =  i['eval_recall']

    raw_pred_train, _, _ = trainer.predict(train_dataset)
    y_pred_train = np.argmax(raw_pred_train, axis=1)
    accuracy = accuracy_score(y_true=y_train, y_pred=y_pred_train)
    history['train_acc'] = accuracy

    if predict == True:
        raw_pred, _, _ = trainer.predict(val_dataset)
        
        # preprocess raw predictions
        y_pred = np.argmax(raw_pred, axis=1)
        report = classification_report(y_val, y_pred, target_names=class_names, digits=4)
        report_path = "./result/report_{}.txt".format(test_count)
        text_file = open(report_path, "w")
        text_file.write(report)

        # copy the best trained model for current test fold
        copytree(f'./output/checkpoint-{best}', f'./result/best_test_{test_count}')

    # clearn the output dictory
    shutil.rmtree('./output', ignore_errors=True)

    return history

if __name__=="__main__":
    pass
