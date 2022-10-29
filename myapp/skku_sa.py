"""
import argparse
import time
import json
import torch
import numpy as np
import pandas as pd
import os
import boto3
import project.settings
from attrdict import AttrDict
from tqdm.notebook import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, AdamW
"""

import json
import os
from attrdict import AttrDict

class SKKU_SENTIMENT:
    def setAttr(self, datapath, modelpath):
        #self.args = loadJSON()
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)
        #self.model = AutoModelForSequenceClassification.from_pretrained(modelpath)
        self.data_path = datapath


    def train(self):
        #self.sentiment_classifier = TextClassificationPipeline(tokenizer=self.tokenizer, model=self.model, function_to_apply=self.args.function_to_apply, device=0)
        pass

    def inference(self):
        pass
        #self.data_path 이용해서 read_csv, list 저장
        #self.inf_result에 list 저장

    def getInfResult(self):
        pass
        #return self.inf_result
  


def loadJSON():
    with open('./config/config.json', encoding="UTF-8") as f:
        args = AttrDict(json.load(f))	
    return args


"""
        # Dataset
        self.data_path = self.args.data_path
        self.dataset = pd.read_csv(self.data_path)
        for idx, val in enumerate(dict(self.args.id2num).items()):
            self.dataset = self.dataset.replace(val[0], idx)
        self.train_data, self.test_data = train_test_split(self.dataset, test_size=self.args.test_size)

        # Train
        self.train()

        # Push to HuggingFace
        self.pushToHub()

        # Save as a local file
        output_dir = os.path.join("samodels", dicname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def train(self):
        for i in range(self.args.epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            batches = 0
            losses = []
            accuracies = []

            train_dataset = SentimentReviewDataset(self.train_data)
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
            optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)

            self.model.train()

            for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader):
				
                optimizer.zero_grad()
                y_batch = list(y_batch)
                for index, i in enumerate(y_batch):
                    if i == '긍정':
                        y_batch[index] = 2
                    elif i == '부정':
                        y_batch[index] = 0
                    else:
                        y_batch[index] = 1
                y_batch = tuple(y_batch)
                y_batch = torch.tensor(y_batch)
                y_batch = y_batch.type(torch.LongTensor)
                print("@@@2")
                y_batch = y_batch.to(self.device)
                y_pred = self.model(input_ids_batch.to(self.device), attention_mask=attention_masks_batch.to(self.device))[0]
                loss = F.cross_entropy(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(y_pred, 1)
                correct += (predicted == y_batch).sum()
                total += len(y_batch)

                batches += 1
                if batches % 100 == 0:
                    print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)

            losses.append(total_loss)
            accuracies.append(correct.float() / total)
            print(i, "Train Loss:", total_loss, "Accuracy:", correct.float() / total)

    
    def evaluate(self):
        start = time.time()
        y_true = list(self.test_data[self.args.data_label[1]])
        y_pred = []
        
        for i in self.test_data.index:
            line = self.test_data.loc[i,self.args.data_label[0]]
            label = self.sentiment_classifier(line)

            for idx, val in enumerate(dict(self.args.label2id).items()):
                if label[0]['label'] == val[0]:
                    y_pred.append(idx)

		
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        target_names = list(self.args.id2num)

        runtime = time.time() - start
        accuracy = accuracy_score(y_true, y_pred)

        print("time :", runtime)
        print("accuracy :", accuracy)
        print(cm_normalized)
        #cmdf = pd.DataFrame(cm,index=self.args.cm_label[0], columns=self.args.cm_label[1])
        #print(cmdf)
        #print(classification_report(y_true, y_pred, target_names=target_names))
    

    def analyze(self, sent:str):
        result = self.sentiment_classifier(sent)
        label_result = result[0]['label']
        sent_result = ''
        
        for idx, val in enumerate(dict(self.args.label2id).items()):
            if label_result == val[0]:
                sent_result = val[1]

        return sent_result


class SentimentReviewDataset(Dataset):  
    def __init__(self, dataset):
        self.args = loadJSON()

        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 0:2].values
        text = row[0]
        y = row[1]

        inputs = self.tokenizer(
            text, 
            return_tensors='pt',
            truncation=True,
            max_length=self.args.max_length,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, y
"""

