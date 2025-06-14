import os
import random
import numpy as np
import torch
import jsonlines
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

METRIC_NAME = 'macro'

def set_random_seed(seed):
    if seed is not None:
        assert seed > 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True # False
        torch.backends.cuda.matmul.allow_tf32 = False # if set it to True will be much faster but not accurate

def read_task_data(data_folder, dimension='readability', choices=['train', 'val', 'test']):
    data_dict = {}
    for choice in choices:
        data_path = os.path.join(data_folder, f'{choice}.jsonl')
        with jsonlines.open(data_path) as f:
            data = [l for l in f]
        data = [{'label': int(item[dimension]), **{k:v for k,v in item.items()}}  for item in data]
        if choice == 'train':
            random.shuffle(data)
        data_dict[choice] = data
    return data_dict

def preprocess_function(examples, tokenizer, max_length=8192):
    result = tokenizer(
        examples['content'],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt"
    )
    return result

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average=METRIC_NAME)
    precision = precision_score(labels, predictions, average=METRIC_NAME)
    recall = recall_score(labels, predictions, average=METRIC_NAME)
    result = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    return result