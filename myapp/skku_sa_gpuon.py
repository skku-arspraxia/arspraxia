import argparse
import time
import json
import torch
import numpy as np
import pandas as pd
import os
import boto3
import project.settings
import csv
import shutil
from .models import NLP_models
from datetime import datetime
from attrdict import AttrDict
from tqdm.notebook import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, AdamW
        
s3c = boto3.client(
    's3',
    aws_access_key_id=project.settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=project.settings.AWS_SECRET_ACCESS_ID
)   
s3r = boto3.resource(
    's3',
    aws_access_key_id=project.settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=project.settings.AWS_SECRET_ACCESS_ID
)

class SKKU_SENTIMENT:
    def setTrainAttr(self, params):
        self.args = loadJSON()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)

        # Hyper Parameters Setting
        self.epochs = int(params["modelepoch"])
        self.batch_size = int(params["modelbs"])
        self.learning_rate = float(params["modellr"])

        # Pretrained Model Download
        localrootPath = "C:/arspraxiabucket/"
        model_path = ""
        modelidx = params["pretrained_model"]
        if int(modelidx) == 0:
            model_path = "monologg/koelectra-small-v3-discriminator"
        else:
            model_path = localrootPath+"model/"+modelidx+"/"  
            modelsrcList = []
            my_bucket = s3r.Bucket('arspraxiabucket')
            for my_bucket_object in my_bucket.objects.all():
                filesrc = my_bucket_object.key.split('.')
                
                if not os.path.exists(model_path):
                    os.makedirs(model_path)

                # 파일 여부 확인
                if len(filesrc) > 1:
                    filepath = filesrc[0].split("/")
                    if filepath[0] == "model" and filepath[1] == modelidx:
                        modelsrcList.append(filesrc[0] + "." + filesrc[1])

            for modelsrc in modelsrcList:
                s3c.download_file('arspraxiabucket', modelsrc, localrootPath+modelsrc)  

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Train Data Download
        tdfilePath = "data/sa/train/"
        tdRemotefileSrc = tdfilePath+params["train_data"]
        tdLocalfilePath = localrootPath+tdfilePath
        tdLocalfileSrc = tdLocalfilePath+params["train_data"]
        if not os.path.exists(tdLocalfilePath):
           os.makedirs(tdLocalfilePath)

        s3c.download_file('arspraxiabucket', tdRemotefileSrc, tdLocalfileSrc)
        self.data_path = tdLocalfileSrc

        fileExtention = params["train_data"].split(".")[1]
        if fileExtention == "tsv":
            try:
                self.dataset = pd.read_csv(self.data_path, encoding="utf-8", delimiter='\t') 
            except:   
                self.dataset = pd.read_csv(self.data_path, encoding="cp949", delimiter='\t')   

        elif fileExtention == "csv": 
            try:
                self.dataset = pd.read_csv(self.data_path, encoding="utf-8") 
            except:   
                self.dataset = pd.read_csv(self.data_path, encoding="cp949")    
                    
        elif fileExtention == "xls" or fileExtention == "xlsx":
                self.dataset = pd.read_excel(self.data_path)  

        for idx, val in enumerate(dict(self.args.id2num).items()):
            self.dataset = self.dataset.replace(val[0], idx)

        self.train_data, self.test_data = train_test_split(self.dataset, test_size=self.args.test_size)


    def train(self):
        for i in range(self.epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            batches = 0
            losses = []
            accuracies = []
            train_dataset = SentimentReviewDataset(self.train_data)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
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
                y_batch = y_batch.to(self.device)
                y_pred = self.model(input_ids_batch, attention_mask=attention_masks_batch)[0].to(self.device)
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
                
        self.sentiment_classifier = TextClassificationPipeline(tokenizer=self.tokenizer, model=self.model, function_to_apply=self.args.function_to_apply, device=0)

        # Save model to local storage
        model_new_idx = 0
        if NLP_models.objects.count() == 0:
            model_new_idx = 1
        else:
            last_data = NLP_models.objects.order_by("id").last()
            model_new_idx = str(last_data.id+1)
        output_dir = "C:/arspraxiabucket/model/"+model_new_idx
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        filelist = ["config.json", "pytorch_model.bin", "training_args.bin"]

        # Get model size
        self.model_size = 0
        for file in filelist:
             self.model_size += os.path.getsize(os.path.join(output_dir, file))

        self.model_size = round(self.model_size/1000000, 2)

        # Save model files in AWS S3 bucket
        for file in filelist:            
            fileuploadname = "model/"+model_new_idx+"/"+file
            with open(os.path.join(output_dir, file), "rb") as f:
                s3c.upload_fileobj(
                        f,
                        'arspraxiabucket',
                        fileuploadname
                )

        # Delete temp local file
        shutil.rmtree(output_dir)

        # Evaluate
        self.evaluate()

    
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
				
        self.runtime = time.time() - start
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precesion = precision_score(y_true, y_pred, average="micro")
        self.recall =recall_score(y_true, y_pred, average="micro")
        self.f1score = f1_score(y_true, y_pred, average="micro")


    def setInferenceAttr(self, params):
        self.args = loadJSON()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)

        # Inference Model Download
        localrootPath = ""
        model_path = ""
        modelidx = params["inference_model"]
        if int(modelidx) == 0:
            model_path = "monologg/koelectra-small-v3-discriminator"
        else:
            localrootPath = "C:/arspraxiabucket/"
            model_path = localrootPath+"model/"+modelidx+"/"  
            modelsrcList = []
            my_bucket = s3r.Bucket('arspraxiabucket')
            for my_bucket_object in my_bucket.objects.all():
                filesrc = my_bucket_object.key.split('.')
                
                if not os.path.exists(model_path):
                    os.makedirs(model_path)

                # 파일 여부 확인
                if len(filesrc) > 1:
                    filepath = filesrc[0].split("/")
                    if filepath[0] == "model" and filepath[1] == modelidx:
                        modelsrcList.append(filesrc[0] + "." + filesrc[1])

            for modelsrc in modelsrcList:
                s3c.download_file('arspraxiabucket', modelsrc, localrootPath+modelsrc)  

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.sentiment_classifier = TextClassificationPipeline(tokenizer=self.tokenizer, model=self.model, function_to_apply=self.args.function_to_apply, device=0)

        # Inference Data Download        
        tdfilePath = "data/sa/inf/"
        tdRemotefileSrc = tdfilePath+params["inference_data"]
        tdLocalfilePath = localrootPath+tdfilePath
        tdLocalfileSrc = tdLocalfilePath+params["inference_data"]
        if not os.path.exists(tdLocalfilePath):
           os.makedirs(tdLocalfilePath)

        s3c.download_file('arspraxiabucket', tdRemotefileSrc, tdLocalfileSrc)
        self.data_path = tdLocalfileSrc


    def inference(self):
        self.inf_result = []
        with open(self.data_path, 'r', encoding="utf-8") as f:
            rdr = csv.reader(f)
            for line in rdr:
                result = self.sentiment_classifier(line[0])
                label_result = result[0]['label']
                sent_result = ''
                
                for idx, val in enumerate(dict(self.args.label2id).items()):
                    if label_result == val[0]:
                        sent_result = val[1]

                self.inf_result.append([line[0], sent_result, 0.77])

        # Save inference result        
        inf_result_path = "C:/arspraxiabucket/data/sa/result/"
        if not os.path.exists(inf_result_path):
            os.makedirs(inf_result_path)
        inf_result_list = self.getInfResult()    
        now = str(datetime.now())
        now = now.replace(':','.')
        now = now[:19]
        filename_str = inf_result_path+now+".csv"
        with open(filename_str, 'w', encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            for i in inf_result_list:
                    wr.writerow(i)


    def getInfResult(self):
        return self.inf_result


    def getModelsize(self):
        return self.model_size


    def getF1score(self):
        return self.f1score


    def getAccuracy(self):
        return self.accuracy


    def getPrecision(self):
        return self.precesion


    def getRecall(self):
        return self.recall


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
            max_length=256,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, y


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

"""