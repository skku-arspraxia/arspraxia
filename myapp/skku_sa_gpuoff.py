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
import pandas as pd
from attrdict import AttrDict
from .models import NLP_models
import os
import boto3
import project.settings
import shutil

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
    def __init__(self):
        self.trainFinished = False
        self.currentStep = 1
        self.currentEpoch = 0

    def setTrainAttr(self, params):
        self.args = loadJSON()
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)

        # Pretrained Model Download
        localrootPath = "C:/arspraxiabucket/"
        model_path = ""
        modelidx = params["pretrained_model"]
        if modelidx == 0:
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

        #self.model = AutoModelForSequenceClassification.from_pretrained(modelpath)

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

        # Save model files in AWS S3 bucket   
        last_data = NLP_models.objects.order_by("id").last() 
        #model_new_idx = str(last_data.id+1)    
        model_new_idx = str(1)    
        output_dir = "C:/arspraxiabucket/model/"+model_new_idx    
        filelist = ["config.json", "pytorch_model.bin", "training_args.bin"]
        for file in filelist:            
            fileuploadname = "model/"+model_new_idx+"/"+file
            with open(os.path.join(output_dir, file), "rb") as f:
                s3c.upload_fileobj(
                        f,
                        'arspraxiabucket',
                        fileuploadname
                )

        # delete downloaded temp model file
        #shutil.rmtree(output_dir)

    def setInferenceAttr(self, params):
        self.args = loadJSON()
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)

        # Inference Model Download
        localrootPath = ""
        model_path = ""
        modelidx = params["inference_model"]
        if modelidx == 0:
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

        #self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        #self.sentiment_classifier = TextClassificationPipeline(tokenizer=self.tokenizer, model=self.model, function_to_apply=self.args.function_to_apply, device=0)

        # Inference Data Download        
        tdfilePath = "data/sa/inf/"
        tdRemotefileSrc = tdfilePath+params["inference_data"]
        tdLocalfilePath = localrootPath+tdfilePath
        tdLocalfileSrc = tdLocalfilePath+params["inference_data"]
        if not os.path.exists(tdLocalfilePath):
           os.makedirs(tdLocalfilePath)

        s3c.download_file('arspraxiabucket', tdRemotefileSrc, tdLocalfileSrc)
        self.data_path = tdLocalfileSrc


    def train(self):
        self.trainFinished = True
        pass
    
    
    def inference(self):
        pass


    def getInfResult(self):
        return 1


    def getModelsize(self):
        return 1


    def getF1score(self):
        return 1


    def getAccuracy(self):
        return 1


    def getPrecision(self):
        return 1


    def getRecall(self):
        return 1

    
    def getCurrentStep(self):
        self.currentStep += 1
        return self.currentStep

    
    def getCurrentEpoch(self):
        self.currentEpoch += 1
        return self.currentEpoch


    def isTrainFinished(self):
        return self.trainFinished


def loadJSON():
    with open('./config/config.json', encoding="UTF-8") as f:
        args = AttrDict(json.load(f))	
    return args

    