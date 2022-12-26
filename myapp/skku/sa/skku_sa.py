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
from ...models import NLP_models
from datetime import datetime
from attrdict import AttrDict
from tqdm.notebook import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
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

class SKKU_SA:

    # class SKKU_SA 시작: args 불러오기, 변수 초기화
    def __init__(self):
        self.args = loadJSON()

    # train 속성 설정
    def setTrainAttr(self, params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)

        # 하이퍼 파라미터 (Epochs, Batch size, Learning rate) 설정
        self.epochs = int(params["modelepoch"])
        self.batch_size = int(params["modelbs"])
        self.learning_rate = float(params["modellr"])

        # pre trained model 다운로드
        localrootPath = self.args.path_local_root
        model_path = ""
        modelidx = params["pretrained_model"]

        # 학습시킬 모델이 이미 finetuning된 모델이라면 AWS S3에서,
        # 그렇지 않다면 Hugging face에서 받아와 모델 저장 (if int(modelidx) == -1)
        if int(modelidx) == -1:
            model_path = self.args.pretrained_model
        else:
            if(int(modelidx) == 0):
                model_path = localrootPath+"model/sa/"  
            else:
                model_path = localrootPath+"model/"+modelidx+"/"  
            modelsrcList = []
            my_bucket = s3r.Bucket(project.settings.AWS_BUCKET_NAME)
            for my_bucket_object in my_bucket.objects.all():
                filesrc = my_bucket_object.key.split('.')
                
                if not os.path.exists(model_path):
                    os.makedirs(model_path)

                # 파일 여부 확인
                if len(filesrc) > 1:
                    filepath = filesrc[0].split("/")
                    if filepath[0] == "model":
                        if int(modelidx) == 0:
                            if filepath[1] == "sa":
                                modelsrcList.append(filesrc[0] + "." + filesrc[1])
                        else:
                            if filepath[1] == modelidx:
                                modelsrcList.append(filesrc[0] + "." + filesrc[1])


            for modelsrc in modelsrcList:
                s3c.download_file(project.settings.AWS_BUCKET_NAME, modelsrc, localrootPath+modelsrc)  
        
        #AutoModelForSequenceClassification 모듈을 통해 Sequence classification pretained 모델을 불러와 저장 
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Train Data Download
        tdfilePath = self.args.path_train_data
        tdRemotefileSrc = tdfilePath+params["train_data"]
        tdLocalfilePath = localrootPath+tdfilePath
        tdLocalfileSrc = tdLocalfilePath+params["train_data"]
        if not os.path.exists(tdLocalfilePath):
           os.makedirs(tdLocalfilePath)

        s3c.download_file(project.settings.AWS_BUCKET_NAME, tdRemotefileSrc, tdLocalfileSrc)
        self.data_path = tdLocalfileSrc

        # “tsv”, “csv”. “xls”, “xlsx” 포맷의 train data를 읽어 데이터 프레임에 저장 
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

    # args의 epochs 수 만큼 for문을 돌며 모델 학습
    def train(self):
        for i in range(self.epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            batches = 0
            losses = []
            accuracies = []
            # class SentimentReviewDataset의 __getitem__ 메소드로 입력받은 train data를 토큰화하여 저장하고 batch size 로 묶어줌
            train_dataset = SentimentReviewDataset(self.train_data)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # optimizer로 AdamW를 사용하여 cross entropy loss값이 최소화되는 방향으로 각 파라미터를 업데이트
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
        # 학습을 마친 모델을 local storage에 저장한 후, 로컬의 모델을 AWS S3에 저장
        model_new_idx = 0
        if NLP_models.objects.count() == 0:
            model_new_idx = 1
        else:
            last_data = NLP_models.objects.order_by("id").last()
            model_new_idx = last_data.id+1
        output_dir = self.args.path_local_root+"model/"+str(model_new_idx)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, self.args.file_args_bin))

        filelist = [self.args.file_config_json, self.args.file_model_bin, self.args.file_args_bin]

        # 모델 사이즈 
        self.model_size = 0
        for file in filelist:
             self.model_size += os.path.getsize(os.path.join(output_dir, file))

        self.model_size = round(self.model_size/1000000, 2)

        # AWS S3 bucket에 모델 파일 저장
        for file in filelist:            
            fileuploadname = "model/"+str(model_new_idx)+"/"+file
            with open(os.path.join(output_dir, file), "rb") as f:
                s3c.upload_fileobj(
                    f,
                    project.settings.AWS_BUCKET_NAME,
                    fileuploadname
                )

        # 임시로 로컬에 저장한 모델 파일 제거
        shutil.rmtree(output_dir)

        # 평가 함수 호출
        self.evaluate()


    # 학습한 모델의 성능평가를 진행하여 Precision, Recall, F1-score의 지표로 표현, 평가에 소요되는 시간 기록
    def evaluate(self):
        start = time.time()
        y_true = list(self.test_data[self.args.data_label[1]]) # test data 의 정답
        y_pred = []  # 모델의 예측 label을 y_pred에 저장
        
        for i in self.test_data.index:
            line = self.test_data.loc[i,self.args.data_label[0]]
            label = self.sentiment_classifier(line)

            for idx, val in enumerate(dict(self.args.label2id).items()):
                if label[0]['label'] == val[0]:
                    y_pred.append(idx)
				
        self.runtime = time.time() - start
        # precision, recall, f1-score 계산
        self.precesion = precision_score(y_true, y_pred, average="micro")
        self.recall =recall_score(y_true, y_pred, average="micro")
        self.f1score = f1_score(y_true, y_pred, average="micro")

    # inference 속성 설정
    def setInferenceAttr(self, params):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)

        # Inference 모델 다운로드
        localrootPath = ""
        model_path = ""
        modelidx = params["inference_model"]

        # 추론을 진행할 모델이 finetuning된 모델이라면 AWS S3에서, 그렇지 않다면 Hugging face에서 
        if int(modelidx) == 0:
            model_path = self.args.pretrained_model
        else:
            localrootPath = self.args.path_local_root
            model_path = localrootPath+"model/"+modelidx+"/"  
            modelsrcList = []
            my_bucket = s3r.Bucket(project.settings.AWS_BUCKET_NAME)
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
                s3c.download_file(project.settings.AWS_BUCKET_NAME, modelsrc, localrootPath+modelsrc)  

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.sentiment_classifier = TextClassificationPipeline(tokenizer=self.tokenizer, model=self.model, function_to_apply=self.args.function_to_apply)

        # 추론을 진행할 unlabeled data 다운로드       
        tdfilePath = self.args.path_inf_data
        tdRemotefileSrc = tdfilePath+params["inference_data"]
        tdLocalfilePath = localrootPath+tdfilePath
        tdLocalfileSrc = tdLocalfilePath+params["inference_data"]
        if not os.path.exists(tdLocalfilePath):
           os.makedirs(tdLocalfilePath)

        s3c.download_file(project.settings.AWS_BUCKET_NAME, tdRemotefileSrc, tdLocalfileSrc)
        self.data_path = tdLocalfileSrc

    # 추론 진행
    def inference(self):
        self.inf_result = []        
        fileExtention = self.data_path.split(".")[1]
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
        # 추론 결과의 label, score (소수점 아래 3자리) 저장
        for line in self.dataset.values.tolist():
            result = self.sentiment_classifier(line[0])
            label_result = result[0]['label']
            sent_result = ''
            score = round(result[0]['score'], 3)

            for idx, val in enumerate(dict(self.args.label2id).items()):
                if label_result == val[0]:
                    sent_result = val[1]

            self.inf_result.append([line[0], sent_result, score])

        # 모델을 통해 추론 진행. 결과 파일은 AWS S3에 추론 완료 시각 이름으로 저장 
        inf_result_path = self.args.path_local_root+self.args.path_result_data
        if not os.path.exists(inf_result_path):
            os.makedirs(inf_result_path)
        inf_result_list = self.getInfResult()    
        now = str(datetime.now())
        now = now.replace(':','')
        now = now.replace('-','')
        now = now.replace(' ','')
        now = now[:14]
        self.result_file_name = now+".csv"
        filename_str = inf_result_path+self.result_file_name

        # 로컬 경로에서 추론 파일 열기
        with open(filename_str, 'w', encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            for i in inf_result_list:
                wr.writerow(i)


        # 추론 결과 aws에 저장   
        fileuploadname = self.args.path_result_data+self.result_file_name
        with open(filename_str, "rb") as f:
            s3c.upload_fileobj(
                f,
                project.settings.AWS_BUCKET_NAME,
                fileuploadname
            )


    def getInfResult(self):
        return self.inf_result


    def getModelsize(self):
        return self.model_size


    def getF1score(self):
        return self.f1score


    def getPrecision(self):
        return self.precesion


    def getRecall(self):
        return self.recall

    
    def getResultFileName(self):
        return self.result_file_name
        

# 감성 분석 데이터 input 형태 맞추기
class SentimentReviewDataset(Dataset):  
    def __init__(self, dataset):
        self.args = loadJSON()
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 0:2].values
        text = row[0] # 문장
        y = row[1]  # label

        inputs = self.tokenizer(
            text, 
            return_tensors=self.args.return_tensors,
            truncation=True,
            max_length=256,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        input_ids = inputs[self.args.input_ids][0]
        attention_mask = inputs[self.args.attention_mask][0]

        return input_ids, attention_mask, y

# JSON 파일에서 args 불러오기
def loadJSON():
    with open(project.settings.SA_JSON_PATH, encoding="UTF-8") as f:
        args = AttrDict(json.load(f))	
    return args