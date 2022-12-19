import os
import copy
import json
import logging
import argparse
import glob
import re
import shutil
import csv 
import boto3
import project.settings
import pandas as pd
import torch
import numpy as np
from ...models import NLP_models
from torch.utils.data import TensorDataset
from datetime import datetime
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict
from sklearn.model_selection import train_test_split

from torch import nn

from transformers.modeling_outputs import TokenClassifierOutput

from typing import Optional, Union, List
from transformers import (
    BasicTokenizer,
    PreTrainedTokenizer,
    Pipeline,
    ModelCard,
    is_tf_available,
    is_torch_available,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification,
    ElectraModel, ElectraPreTrainedModel
)

from transformers.pipelines import ArgumentHandler
from myapp.skku.ner.utils import set_seed
from myapp.skku.ner.utils import CONFIG_CLASSES
from myapp.skku.ner.utils import TOKENIZER_CLASSES
from myapp.skku.ner.utils import MODEL_FOR_TOKEN_CLASSIFICATION
from myapp.skku.ner.utils import compute_metrics
from myapp.skku.ner.utils import show_ner_report
        
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
class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """
 
    def __init__(self, words):
        self.words = words
 
    def __repr__(self):
        return str(self.to_json_string())
 
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
 
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
 
 
class InputFeatures(object):
    """A single set of features of data."""
 
    def __init__(self, input_ids, attention_mask, crf_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.crf_mask = crf_mask
 
    def __repr__(self):
        return str(self.to_json_string())
 
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
 
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
 
 
class NerProcessor(object):
    def __init__(self, text):
        self.text = text
 
    def get_labels(self):
        return ["O",
                "PER-B", "PER-I", "ORG-B", "ORG-I",
                "LOC-B", "LOC-I", "ISSUE-B", "ISSUE-I"]
 
    @classmethod
    def _strip(cls, input_text):
        for i in range(len(input_text)):
            input_text[i] = input_text[i].strip()
        return input_text
 
    def _create_examples(self, text):
        examples = []
        for (i, sentence) in enumerate(text):
            words = sentence.split()
            examples.append(InputExample(words=words, labels=[]))
        return examples
 
    def get_examples(self):
        return self._create_examples(self._strip(self.text))


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    features = []
    for (ex_idx, example) in enumerate(examples):
        tokens = []
        crf_mask = []
 
        for word in example.words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]
            tokens.extend(word_tokens)
            crf_mask.extend([True] + [False] * (len(word_tokens) - 1))
 
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            crf_mask = crf_mask[:(max_seq_length - special_tokens_count)]
 
        # Add [SEP]
        tokens += [tokenizer.sep_token]
        crf_mask += [False]
 
        # Add [CLS]
        tokens = [tokenizer.cls_token] + tokens
        crf_mask = [False] + crf_mask
 
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
 
        attention_mask = [1] * len(input_ids)
 
        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        crf_mask += [False] * padding_length
 
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(crf_mask) == max_seq_length
 
        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          crf_mask=crf_mask,
                          token_type_ids=input_ids,
                          label_ids="",
                          label_mask="")
        )
 
    return features
 
def make_examples(tokenizer, text):
    MAX_SEQ_LEN = 128
    processor = NerProcessor(text)
    examples = processor.get_examples()
    features = convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length=MAX_SEQ_LEN
    )
 
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_crf_mask = torch.tensor([f.crf_mask for f in features], dtype=torch.bool)
 
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_crf_mask)
    return dataset
 



class SKKU_NER_CRF:    
    def __init__(self):
        self.trainFinished = False
        self.currentStep = 1
        self.currentEpoch = 0
        self.f1score = 0
        self.precesion = 0
        self.recall = 0


    def setTrainAttr(self, params):
        self.args = loadJSON()
        set_seed(self.args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TOKENIZER_CLASSES[self.args.model_type].from_pretrained(
            self.args.model_name_or_path,
            do_lower_case=self.args.do_lower_case
        )
        
        # Hyper Parameters Setting
        self.epochs = int(params["modelepoch"])
        self.batch_size = int(params["modelbs"])
        self.learning_rate = float(params["modellr"])

        # Pretrained Model Download
        localrootPath = self.args.path_local_root
        model_path = ""
        modelidx = params["pretrained_model"]
        if int(modelidx) == -1:
            # Hugging Face
            processor = NaverNerProcessor(self.args)
            labels = processor.get_labels()
            config = CONFIG_CLASSES[self.args.model_type].from_pretrained(
                self.args.model_name_or_path,
                num_labels=self.args.num_labels,
                id2label={str(i): label for i, label in enumerate(labels)},
                label2id={label: i for i, label in enumerate(labels)},
            )
            self.model = KoelectraCRF.from_pretrained(
                self.args.model_name_or_path,
                config=config
            )
        else:
            # Local
            if(int(modelidx) == 0):
                model_path = localrootPath+"model/ner/"  
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
                            if filepath[1] == "ner":
                                modelsrcList.append(filesrc[0] + "." + filesrc[1])
                        else:
                            if filepath[1] == modelidx:
                                modelsrcList.append(filesrc[0] + "." + filesrc[1])

            for modelsrc in modelsrcList:
                s3c.download_file(project.settings.AWS_BUCKET_NAME, modelsrc, localrootPath+modelsrc)  

            self.model = KoelectraCRF.from_pretrained(model_path)

        self.currentStep = 2    # Finished downloading model
        self.model.to(self.device)

        # Train Data Download
        tdfilePath = self.args.path_train_data
        tdRemotefileSrc = tdfilePath+params["train_data"]
        tdLocalfilePath = localrootPath+tdfilePath
        tdLocalfileSrc = tdLocalfilePath+params["train_data"]
        if not os.path.exists(tdLocalfilePath):
           os.makedirs(tdLocalfilePath)

        s3c.download_file(project.settings.AWS_BUCKET_NAME, tdRemotefileSrc, tdLocalfileSrc)
        self.data_path = tdLocalfileSrc
        
        # Load dataset
        self.train_dataset = load_and_cache_examples(self.args, self.tokenizer, self.data_path)
        self.test_dataset = None
        self.dev_dataset = None
        
        self.train_dataset, self.test_dataset = train_test_split(self.train_dataset, test_size=self.args.test_size, random_state=1)
        self.currentStep = 3    # Finished downloading data


    def train(self):
        self.check_is_not_exist = True

        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.epochs
    
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        electra_param_optimizer = list(self.model.electra.named_parameters())
        crf_param_optimizer = list(self.model.crf.named_parameters())
        linear_param_optimizer = list(self.model.position_wise_ff.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in electra_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate},
            {'params': [p for n, p in electra_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': self.args.learning_rate},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay, 'lr': self.args.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': self.args.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay, 'lr': self.args.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': self.args.crf_learning_rate}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.args.warmup_proportion), num_training_steps=t_total)
    
        if os.path.isfile(os.path.join(self.args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(self.args.model_name_or_path, "scheduler.pt")
        ):
            # Load optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "scheduler.pt")))
    
        # Train!
        self.args.save_steps = t_total
        self.args.logging_steps = t_total/self.epochs
        
        global_step = 0
        tr_loss = 0.0

        self.model.zero_grad()
        mb = master_bar(range(int(self.epochs)))
        for epoch in mb:
            epoch_iterator = progress_bar(train_dataloader, parent=mb)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                    "label_mask": batch[4],
                    "crf_mask": batch[5]
                }
                if self.args.model_type not in ["distilkobert", "xlm-roberta"]:
                    inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            
                outputs = self.model(**inputs)
                loss = outputs['loss']
    
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
    
                loss.backward()
                tmp_loss = loss.item()
                tr_loss += tmp_loss
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        len(train_dataloader) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(train_dataloader)
                ):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
    
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    break

            mb.write("Epoch {} done".format(epoch + 1))
            self.currentEpoch += 1

            if self.args.max_steps > 0 and global_step > self.args.max_steps:
                break
        self.currentStep = 4    # Finished training

        # Save model to local storage
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

        # Get model size
        self.model_size = 0
        for file in filelist:
            self.model_size += os.path.getsize(os.path.join(output_dir, file))

        self.model_size = round(self.model_size/1000000, 2)

        # Save model files in AWS S3 bucket
        for file in filelist:            
            fileuploadname = "model/"+str(model_new_idx)+"/"+file
            with open(os.path.join(output_dir, file), "rb") as f:
                s3c.upload_fileobj(
                    f,
                    project.settings.AWS_BUCKET_NAME,
                    fileuploadname
                )

        # Delete temp local file
        shutil.rmtree(output_dir)
        self.currentStep = 5    # Finished uploading model
        
        self.evaluate()

    
    def evaluate(self):
        results = {}
        eval_sampler = SequentialSampler(self.test_dataset)
        eval_dataloader = DataLoader(self.test_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
    
        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        label_mask = None

        for batch in progress_bar(eval_dataloader):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                    "label_mask" : batch[4],
                    "crf_mask" : batch[5]
                }
                if self.args.model_type not in ["distilkobert", "xlm-roberta"]:
                    inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
                outputs = self.model(**inputs)
                tmp_eval_loss = outputs['loss']
                tags = outputs['y_pred']
                
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = tags
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                label_mask = inputs["label_mask"].detach().cpu().numpy()
            else:
                for tag in tags :
                    preds.append(tag)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                label_mask = np.append(label_mask, inputs["label_mask"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        ner_processor = NaverNerProcessor(self.args)
        labels = ner_processor.get_labels()

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        pad_token_label_id = CrossEntropyLoss().ignore_index

        
        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if label_mask[i,j] == 1:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])

        for seq_idx, pred in enumerate(preds) :
            for j in pred :
                preds_list[seq_idx].append(label_map[j])

        result = compute_metrics(self.args.task, out_label_list, preds_list)
        results.update(result)

        self.f1score = results["f1"]
        self.precision = results["precision"]
        self.recall = result["recall"]
    

    def setInferenceAttr(self, params):   
        self.args = loadJSON()
        set_seed(self.args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TOKENIZER_CLASSES[self.args.model_type].from_pretrained(
            self.args.model_name_or_path,
            do_lower_case=self.args.do_lower_case
        )
        # Inference Model Download
        localrootPath = self.args.path_local_root
        model_path = ""
        modelidx = params["inference_model"]
        if int(modelidx) == 0:
            # Hugging Face
            processor = NaverNerProcessor(self.args)
            labels = processor.get_labels()
            config = CONFIG_CLASSES[self.args.model_type].from_pretrained(
                self.args.model_name_or_path,
                num_labels=self.args.num_labels,
                id2label={str(i): label for i, label in enumerate(labels)},
                label2id={label: i for i, label in enumerate(labels)},
            )
            self.model = KoelectraCRF.from_pretrained(
                self.args.model_name_or_path,
                config=config
            )
        else:
            # Local
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
 
            self.model = KoelectraCRF.from_pretrained(model_path)

        #############################################
        # Inference Data Download  
        localrootPath = self.args.path_local_root      
        tdfilePath = self.args.path_inf_data
        tdRemotefileSrc = tdfilePath+params["inference_data"]
        tdLocalfilePath = localrootPath+tdfilePath
        tdLocalfileSrc = tdLocalfilePath+params["inference_data"]
        if not os.path.exists(tdLocalfilePath):
           os.makedirs(tdLocalfilePath)
        s3c.download_file(project.settings.AWS_BUCKET_NAME, tdRemotefileSrc, tdLocalfileSrc)
        self.data_path = tdLocalfileSrc

    def inference(self):
        
        text=[]
        fileExtention = self.data_path.split(".")[1]
        if fileExtention == "xls" or fileExtention == "xlsx":
            exellines = pd.read_excel(self.data_path, usecols=[0])
            for line in exellines.values.tolist():
                if type(line) is str:
                    tline = line.strip()
                    text.append(tline)
                else:
                    tline = line[0].strip()
                    text.append(tline)
        elif fileExtention == "tsv":
            with open(self.data_path,'r', encoding="utf-8") as read:
                lines=csv.reader(read, delimiter="\t")
                for line in lines:
                    tline = line[0].strip()
                    text.append(tline)
        else:
            with open(self.data_path,'r', encoding="utf-8") as read:
                lines=csv.reader(read)
                for line in lines:
                    tline = line[0].strip()
                    text.append(tline)

        self.model.to(self.device)
        processor=NerProcessor(text)
        labels=processor.get_labels()
       
        str_dataset=make_examples(self.tokenizer, text)
        str_sampler=SequentialSampler(str_dataset)
        str_dataloader=DataLoader(str_dataset,sampler=str_sampler,batch_size=len(text))
        preds=None
        #label_mask=None ?
        for batch in str_dataloader:
            self.model.eval()
            batch=tuple(t.to(self.device) for t in batch)
           
            with torch.no_grad():
                inputs={
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "crf_mask":batch[2]
                }
                outputs=self.model(**inputs)
                tags=outputs['y_pred']
            preds=tags
            #label_mask=inputs["crf_mask"].detach().cpu().numpy() ?

        label_map={i: label for i, label in enumerate(labels)}
       
        preds_list = [[] for _ in range(len(preds))]
       
        for seq_idx, pred in enumerate(preds):
            for j in pred :
                preds_list[seq_idx].append(label_map[j])
         
        #self.ner = NerPipeline(model=self.model, tokenizer=self.tokenizer, ignore_labels=[], ignore_special_tokens=True)  
       
        inf_result_path = self.args.path_local_root+self.args.path_result_data
        if not os.path.exists(inf_result_path):
            os.makedirs(inf_result_path)  
        now = str(datetime.now())
        now = now.replace(':','')
        now = now.replace('-','')
        now = now.replace(' ','')
        now = now[:14]
        self.result_file_name = now+".csv"
        outputpath =  inf_result_path+self.result_file_name
        with open(outputpath,'w', encoding="utf-8") as write:
            wr=csv.writer(write)
            i=0
            for line in text:
                predstr = ' '.join(s for s in preds_list[i])
                wr.writerow([line,predstr])
                i+=1
        #if excel
        # f_csv=pd.read_csv(outputpath)
        # f_excel=pd.ExcelWriter(now+".xlsx")
        # f_csv.to_excel(f_excel, index=False)
        # f_excel.save()
        # outputpath=f_excel
        ###            
       
        fileuploadname = self.args.path_result_data+self.result_file_name
        with open(outputpath, "rb") as f:
            s3c.upload_fileobj(
                f,
                project.settings.AWS_BUCKET_NAME,
                fileuploadname
            )
    
    def getModelsize(self):
        return self.model_size


    def getF1score(self):
        return self.f1score


    def getPrecision(self):
        return self.precesion


    def getRecall(self):
        return self.recall

    
    def getCurrentStep(self):
        return self.currentStep

    
    def getCurrentEpoch(self):
        return self.currentEpoch

    
    def getResultFileName(self):
        return self.result_file_name


    def isTrainFinished(self):
        return self.trainFinished


class InputExample(object):
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids, label_mask, crf_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids
        self.label_mask = label_mask
        self.crf_mask = crf_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def ner_convert_examples_to_features(
        args,
        examples,
        tokenizer,
        max_seq_length,
        task,
        pad_token_label_id=0,
        bos_token_label_id=0,
        eos_token_label_id=0,
):
    label_lst = NaverNerProcessor(args)
    label_lst_label = label_lst.get_labels()
    label_map = {label: i for i, label in enumerate(label_lst_label)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = []
        label_ids = []
        label_mask = []
        crf_mask = []

        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if label.endswith("-B"):
                label_ids.extend([label_map[label]] + [label_map[label] + 1] * (len(word_tokens) - 1))
            else :
                label_ids.extend([label_map[label]] + [label_map[label]] * (len(word_tokens) - 1))

            label_mask.extend([1] + [0] * (len(word_tokens) - 1))
            crf_mask.extend([True] + [False] * (len(word_tokens) - 1))
        
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]
            label_mask = label_mask[:(max_seq_length - special_tokens_count)]
            crf_mask = crf_mask[:(max_seq_length - special_tokens_count)]

        # Add [SEP]
        tokens += [tokenizer.sep_token]
        label_ids += [pad_token_label_id]
        label_mask += [0]
        crf_mask += [False]

        # Add [CLS]
        tokens = [tokenizer.cls_token] + tokens
        label_ids = [bos_token_label_id] + label_ids
        label_mask = [0] + label_mask
        crf_mask = [False] + crf_mask

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        label_mask += [pad_token_label_id] * padding_length
        crf_mask += [False] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length
        assert len(crf_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_ids=label_ids,
                          label_mask=label_mask,
                          crf_mask=crf_mask)
        )
    return features

class NaverNerProcessor(object):
    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["O",
                "PER-B", "PER-I", "ISSUE-B", "ISSUE-I", "ORG-B", "ORG-I",
                "LOC-B", "LOC-I"]

    def _create_examples(self, input_file):
        examples = []
        fileExtention = input_file.split(".")[1]
        if fileExtention == "tsv":
            try:
                self.dataset = pd.read_csv(input_file, encoding="utf-8", delimiter='\t') 
            except:   
                self.dataset = pd.read_csv(input_file, encoding="cp949", delimiter='\t')   

        elif fileExtention == "csv": 
            try:
                self.dataset = pd.read_csv(input_file, encoding="utf-8") 
            except:   
                self.dataset = pd.read_csv(input_file, encoding="cp949")    
                    
        elif fileExtention == "xls" or fileExtention == "xlsx":
                self.dataset = pd.read_excel(input_file)

        for words, labels in self.dataset.values.tolist():
            words = words.split()
            labels = labels.split()
            examples.append(InputExample(words=words, labels=labels))
        return examples


    def get_examples(self, data_path):
        return self._create_examples(data_path)


def load_and_cache_examples(args, tokenizer, data_path):
    processor = NaverNerProcessor(args)
    examples = processor.get_examples(data_path)
    
    features = ner_convert_examples_to_features(
        args,
        examples,
        tokenizer,
        max_seq_length=args.max_seq_len,
        task=args.task
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    all_crf_mask = torch.tensor([f.crf_mask for f in features], dtype=torch.bool)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, all_label_mask, all_crf_mask)

    return dataset


def custom_encode_plus(sentence,
                       tokenizer,
                       return_tensors=None):
    # {'input_ids': [2, 10841, 10966, 10832, 10541, 21509, 27660, 18, 3], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0]}
    words = sentence.split()

    tokens = []
    tokens_mask = []

    for word in words:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
        tokens.extend(word_tokens)
        tokens_mask.extend([1] + [0] * (len(word_tokens) - 1))

    ids = tokenizer.convert_tokens_to_ids(tokens)
    len_ids = len(ids)
    total_len = len_ids + tokenizer.num_special_tokens_to_add()
    """ if tokenizer.max_len and total_len > tokenizer.max_len:
        ids, _, _ = tokenizer.truncate_sequences(
            ids,
            pair_ids=None,
            num_tokens_to_remove=total_len - tokenizer.max_len,
            truncation_strategy="longest_first",
            stride=0,
        ) """

    sequence = tokenizer.build_inputs_with_special_tokens(ids)
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(ids)
    # HARD-CODED: As I know, most of the transformers architecture will be `[CLS] + text + [SEP]``
    #             Only way to safely cover all the cases is to integrate `token mask builder` in internal library.
    tokens_mask = [1] + tokens_mask + [1]
    words = [tokenizer.cls_token] + words + [tokenizer.sep_token]

    encoded_inputs = {}
    encoded_inputs["input_ids"] = sequence
    encoded_inputs["token_type_ids"] = token_type_ids

    if return_tensors == "tf" and is_tf_available():
        encoded_inputs["input_ids"] = tf.constant([encoded_inputs["input_ids"]])

        if "token_type_ids" in encoded_inputs:
            encoded_inputs["token_type_ids"] = tf.constant([encoded_inputs["token_type_ids"]])

        if "attention_mask" in encoded_inputs:
            encoded_inputs["attention_mask"] = tf.constant([encoded_inputs["attention_mask"]])

    elif return_tensors == "pt" and is_torch_available():
        encoded_inputs["input_ids"] = torch.tensor([encoded_inputs["input_ids"]])

        if "token_type_ids" in encoded_inputs:
            encoded_inputs["token_type_ids"] = torch.tensor([encoded_inputs["token_type_ids"]])

        if "attention_mask" in encoded_inputs:
            encoded_inputs["attention_mask"] = torch.tensor([encoded_inputs["attention_mask"]])

    return encoded_inputs, words, tokens_mask


class NerPipeline(Pipeline):
    default_input_names = "sequences"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        ignore_labels=["O"],
        task: str = "",
        ignore_special_tokens: bool = True
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task,
        )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self.ignore_labels = ignore_labels
        self.ignore_special_tokens = ignore_special_tokens

    def __call__(self, *texts, **kwargs):
        inputs = [*texts]
        answers = []
        for sentence in inputs:

            # Manage correct placement of the tensors
            with self.device_placement():

                # [FIX] Split token by word-level
                tokens, words, tokens_mask = custom_encode_plus(
                    sentence,
                    self.tokenizer,
                    return_tensors=self.framework
                )

                # Forward
                if self.framework == "tf":
                    entities = self.model(tokens)[0][0].numpy()
                    input_ids = tokens["input_ids"].numpy()[0]
                else:
                    with torch.no_grad():
                        tokens = self.ensure_tensor_on_device(**tokens)
                        entities = self.model(**tokens)[0][0].cpu().numpy()
                        input_ids = tokens["input_ids"].cpu().numpy()[0]

            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)

            token_level_answer = []
            for idx, label_idx in enumerate(labels_idx):
                # NOTE Append every answer even though the `entity` is in `ignore_labels`
                token_level_answer += [
                    {
                        "word": self.tokenizer.convert_ids_to_tokens(int(input_ids[idx])),
                        "score": score[idx][label_idx].item(),
                        "entity": self.model.config.id2label[label_idx],
                    }
                ]

            # [FIX] Now let's change it to word-level NER
            word_idx = 0
            word_level_answer = []

            # NOTE: Might not be safe. BERT, ELECTRA etc. won't make issues.
            if self.ignore_special_tokens:
                words = words[1:-1]
                tokens_mask = tokens_mask[1:-1]
                token_level_answer = token_level_answer[1:-1]

            for mask, ans in zip(tokens_mask, token_level_answer):
                if mask == 1:
                    ans["word"] = words[word_idx]
                    word_idx += 1
                    if ans["entity"] not in self.ignore_labels:
                        word_level_answer.append(ans)

            # Append
            answers += [word_level_answer]
        if len(answers) == 1:
            return answers[0]
        return answers

        
    def _sanitize_parameters(self, **pipeline_parameters):
        return None, None, None

    def preprocess(self, **pipeline_parameters):
        pass

    def postprocess(self, **pipeline_parameters):
        pass
    
    def _forward(self, **pipeline_parameters):
        pass

class KoelectraCRF(ElectraPreTrainedModel):
    def __init__(self, config):
        super(KoelectraCRF, self).__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_wise_ff = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, label_mask=None, crf_mask=None):
        outputs =self.electra(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.position_wise_ff(sequence_output)
        # outputs = (logits,)
        outputs = {}
        outputs['logits'] = logits

        mask = crf_mask
        batch_size = logits.shape[0]

        if labels is not None:
            loss = 0
            for seq_logits, seq_labels, seq_mask in zip(logits, labels, mask):
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(emissions = seq_logits, tags=seq_labels, reduction='token_mean')
            loss /= batch_size
            outputs['loss'] = loss

        # else :
        #     output_tags = []
        #     for seq_logits, seq_mask in zip(logits, mask):
        #         seq_logits = seq_logits[seq_mask].unsqueeze(0)
        #         tags = self.crf.decode(seq_logits)
        #         output_tags.append(tags[0])
        #     outputs['y_pred'] = output_tags
        output_tags = []
        for seq_logits, seq_mask in zip(logits, mask):
            seq_logits = seq_logits[seq_mask].unsqueeze(0)
            tags = self.crf.decode(seq_logits)
            output_tags.append(tags[0])
        outputs['y_pred'] = output_tags

        return outputs



class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()


    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)


    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)

            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions

            next_score = torch.logsumexp(next_score, dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)

            broadcast_emission = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emission

            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):

            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

def loadJSON():
    with open("./config/config_ner_crf.json", encoding="UTF-8") as f:
        args = AttrDict(json.load(f))	
    return args
