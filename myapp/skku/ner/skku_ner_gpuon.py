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

from typing import Optional, Union
from transformers import (
    BasicTokenizer,
    PreTrainedTokenizer,
    Pipeline,
    ModelCard,
    is_tf_available,
    is_torch_available,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
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

class SKKU_NER:    
    def __init__(self):
        self.trainFinished = False
        self.currentStep = 1
        self.currentEpoch = 0
        self.f1score = 0
        self.accuracy = 0
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
        localrootPath = "C:/arspraxiabucket/"
        model_path = ""
        modelidx = params["pretrained_model"]
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
            self.model = MODEL_FOR_TOKEN_CLASSIFICATION[self.args.model_type].from_pretrained(
                self.args.model_name_or_path,
                config=config
            )
        else:
            # Local
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

            self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        self.currentStep = 2    # Finished downloading model
        self.model.to(self.device)

        # Train Data Download
        tdfilePath = "data/ner/train/"
        tdRemotefileSrc = tdfilePath+params["train_data"]
        tdLocalfilePath = localrootPath+tdfilePath
        tdLocalfileSrc = tdLocalfilePath+params["train_data"]
        if not os.path.exists(tdLocalfilePath):
           os.makedirs(tdLocalfilePath)

        s3c.download_file('arspraxiabucket', tdRemotefileSrc, tdLocalfileSrc)
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
                    "labels": batch[3]
                }
                if self.args.model_type not in ["distilkobert", "xlm-roberta"]:
                    inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
                outputs = self.model(**inputs)
    
                loss = outputs[0]
    
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
    
                loss.backward()
                tr_loss += loss.item()
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
        output_dir = "C:/arspraxiabucket/model/"+str(model_new_idx)
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
            fileuploadname = "model/"+str(model_new_idx)+"/"+file
            with open(os.path.join(output_dir, file), "rb") as f:
                s3c.upload_fileobj(
                        f,
                        'arspraxiabucket',
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

        for batch in progress_bar(eval_dataloader):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3]
                }
                if self.args.model_type not in ["distilkobert", "xlm-roberta"]:
                    inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        preds = np.argmax(preds, axis=2)

        ner_processor = NaverNerProcessor(self.args)
        labels = ner_processor.get_labels()

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        pad_token_label_id = CrossEntropyLoss().ignore_index

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        result = compute_metrics(self.args.task, out_label_list, preds_list)
        results.update(result)

        #
        # f1, precision, recall 저장
        self.f1score = results["f1"]
        self.accuracy = results["accuracy"]
        self.precision = results["precision"]
        self.recall = result["recall"]
        #
    
    def setInferenceAttr(self, params):        
        self.args = loadJSON()
        set_seed(self.args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TOKENIZER_CLASSES[self.args.model_type].from_pretrained(
            self.args.model_name_or_path,
            do_lower_case=self.args.do_lower_case
        )

        # Inference Model Download
        localrootPath = "C:/arspraxiabucket/"
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
            self.model = MODEL_FOR_TOKEN_CLASSIFICATION[self.args.model_type].from_pretrained(
                self.args.model_name_or_path,
                config=config
            )
        else:
            # Local
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

            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
      
        self.ner = NerPipeline(model=self.model, tokenizer=self.tokenizer, ignore_labels=[], ignore_special_tokens=True)   
        
        # Inference Data Download  
        localrootPath = "C:/arspraxiabucket/"      
        tdfilePath = "data/ner/inf/"
        tdRemotefileSrc = tdfilePath+params["inference_data"]
        tdLocalfilePath = localrootPath+tdfilePath
        tdLocalfileSrc = tdLocalfilePath+params["inference_data"]
        if not os.path.exists(tdLocalfilePath):
           os.makedirs(tdLocalfilePath)
        s3c.download_file('arspraxiabucket', tdRemotefileSrc, tdLocalfileSrc)
        self.data_path = tdLocalfileSrc


    def inference(self):
        # Save inference result      
        inputfile = open(self.data_path, 'r', encoding="utf-8")  
        inf_result_path = "C:/arspraxiabucket/data/ner/result/"
        if not os.path.exists(inf_result_path):
            os.makedirs(inf_result_path)  
        now = str(datetime.now())
        now = now.replace(':','.')
        now = now[:19]
        outputpath =  inf_result_path+now+".csv"
        outputfile = open(outputpath, 'w', encoding='utf-8', newline='')

        wr = csv.writer(outputfile)
        lines = inputfile.readlines()
        for line in lines:
            ners = ""
            anal = self.ner(line)
            for i in anal:
                ners = ners + i['entity'] + ' '
                ners = ners[:-1]
            wr.writerow([line, ners])
        inputfile.close()
        outputfile.close()

    
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

    
    def getCurrentStep(self):
        return self.currentStep

    
    def getCurrentEpoch(self):
        return self.currentEpoch


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
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

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
        pad_token_label_id=-100,
):
    label_lst = NaverNerProcessor(args)
    label_lst_label = label_lst.get_labels()
    label_map = {label: i for i, label in enumerate(label_lst_label)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = []
        label_ids = []

        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # Add [SEP]
        tokens += [tokenizer.sep_token]
        label_ids += [pad_token_label_id]

        # Add [CLS]
        tokens = [tokenizer.cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_ids=label_ids)
        )
    return features

class NaverNerProcessor(object):
    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["O",
                "PER-B", "PER-I", "ISSUE-B", "ISSUE-I", "ORG-B", "ORG-I",
                "LOC-B", "LOC-I"]

    @classmethod
    def _read_file(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, dataset):
        examples = []
        for (i, data) in enumerate(dataset):
            words, labels = data.split('\t')
            words = words.split()
            labels = labels.split()
            assert len(words) == len(labels)

            examples.append(InputExample(words=words, labels=labels))
        return examples

    def get_examples(self, data_path):
        return self._create_examples(self._read_file(data_path))


def load_and_cache_examples(args, tokenizer, data_path):
    processor = NaverNerProcessor(args)
    examples = processor.get_examples(data_path)
    
    pad_token_label_id = CrossEntropyLoss().ignore_index
    features = ner_convert_examples_to_features(
        args,
        examples,
        tokenizer,
        max_seq_length=args.max_seq_len,
        task=args.task,
        pad_token_label_id=pad_token_label_id
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)

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


def loadJSON():
    with open("./config/config_ner.json", encoding="UTF-8") as f:
        args = AttrDict(json.load(f))	
    return args
