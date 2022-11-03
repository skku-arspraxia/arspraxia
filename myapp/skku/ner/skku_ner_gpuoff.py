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

from typing import Optional, Union
from attrdict import AttrDict
from datetime import datetime
"""
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from fastprogress.fastprogress import master_bar, progress_bar
from sklearn.model_selection import train_test_split
"""
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
        pass


    def train(self):
        pass
    
    def evaluate(self):
        pass
        #
        # f1, precision, recall 저장
        # self.f1score = 0
        # self.accuracy = 0
        # self.precesion = 0
        # self.recall = 0
        #
    
    def setInferenceAttr(self, params):        
        pass


    def inference(self):
        pass

    
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
    label_lst = NaverNerProcessor.get_labels()
    label_map = {label: i for i, label in enumerate(label_lst)}

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
    if tokenizer.max_len and total_len > tokenizer.max_len:
        ids, _, _ = tokenizer.truncate_sequences(
            ids,
            pair_ids=None,
            num_tokens_to_remove=total_len - tokenizer.max_len,
            truncation_strategy="longest_first",
            stride=0,
        )

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
        inputs = self._args_parser(*texts, **kwargs)
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


def loadJSON():
    with open("./nermodel_before_train/config_ner.json", encoding="UTF-8") as f:
        args = AttrDict(json.load(f))	
    return args
