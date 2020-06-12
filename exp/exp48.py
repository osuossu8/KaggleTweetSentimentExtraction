import numpy as np
import pandas as pd
import argparse
import collections
import cv2
import datetime
import gc
import glob
import logging
import math
import operator
import os 
import pickle
import pkg_resources
import random
import re
import scipy.stats as stats
import seaborn as sns
import shutil
import string
import sys
import time
import torch
import tokenizers
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import transformers
from contextlib import contextmanager
from collections import OrderedDict
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_log_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.nn import CrossEntropyLoss, MSELoss
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR
from torch.utils import model_zoo
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import tensorflow as tf

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
# from apex import amp

sys.path.append("/usr/src/app/kaggle/tweet-sentiment-extraction")

EXP_ID = "exp48"
import src.configs.config48 as config
import src.engine48 as engine
from src.machine_learning_util import seed_everything, prepare_labels, timer, to_pickle, unpickle
from src.model2 import TweetRoBERTaModel, TweetRoBERTaModelSimple, TweetRoBERTaModelConv1dHead, TweetRoBERTaModelConv1dHeadV2


#import io
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


SEED = 718
seed_everything(SEED)


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


LOGGER_PATH = f"logs/log_{EXP_ID}.txt"
setup_logger(out_file=LOGGER_PATH)
LOGGER.info("seed={}".format(SEED))


def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    
    input_ids = np.ones(max_len, dtype='int32')
    attention_mask = np.zeros(max_len, dtype='int32')
    token_type_ids = np.zeros(max_len, dtype='int32')
    start_tokens = np.zeros(max_len, dtype='int32')
    end_tokens = np.zeros(max_len, dtype='int32')
    
    text1 = " " + " ".join(str(tweet).split())
    text2 = " ".join(str(selected_text).split())  
    
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx+len(text2)]=1
    if text1[idx-1]==' ': chars[idx-1] = 1 
    enc = tokenizer.encode(text1)

    offsets = []; idx=0
    for t in enc.ids:
        w = tokenizer.decode([t])
        offsets.append((idx,idx+len(w)))
        idx += len(w)

    toks = []
    for i,(a,b) in enumerate(offsets):
        sm = np.sum(chars[a:b])
        if sm>0: toks.append(i) 
        
    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }   
        
    s_tok = sentiment_id[sentiment]
    input_ids[:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]
    attention_mask[:len(enc.ids)+3] = 1
    if len(toks)>0:
        start_tokens[toks[0]+2] = 1
        end_tokens[toks[-1]+2] = 1
        
   
    return {
        'ids': input_ids,
        'mask': attention_mask,
        'token_type_ids': token_type_ids,
        'targets_start': np.argmax(start_tokens),
        'targets_end': np.argmax(end_tokens),
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
    }


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
        }


def remove_urls(x):
    x = str(x) 
    x = re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "" ,x)
    return x


def under_score_processing(x):
    if str(x).startswith('_'):
        return x.replace(x.split()[0], '')
    return x


def run_one_fold(fold_id):

    with timer('load csv data'):

        debug = config.DEBUG
        df_train = pd.read_csv(config.TRAIN_PATH).dropna().reset_index(drop=True)

        if debug:
            df_train = df_train.sample(1000, random_state=SEED).dropna().reset_index(drop=True)

        # df_train['text'] = df_train['text'].map(remove_urls)
        # df_train['selected_text'] = df_train['selected_text'].map(remove_urls)
        # df_train['text'] = df_train['text'].apply(lambda x: x.replace('_', ' '))
        # df_train['selected_text'] = df_train['selected_text'].apply(lambda x: x.replace('_', ' '))

        #df_train['text'] = df_train['text'].apply(lambda x: under_score_processing(x))
        #df_train['selected_text'] = df_train['selected_text'].apply(lambda x: under_score_processing(x))

        num_folds = config.NUM_FOLDS
        kf = StratifiedKFold(n_splits = num_folds, random_state = SEED)
        splits = list(kf.split(X=df_train, y=df_train[['sentiment']]))
        train_idx = splits[fold_id][0]
        val_idx = splits[fold_id][1]

        print(len(train_idx), len(val_idx))

        gc.collect()


    with timer('prepare validation data'):
        train_dataset = TweetDataset(
            tweet=df_train.iloc[train_idx].text.values,
            sentiment=df_train.iloc[train_idx].sentiment.values,
            selected_text=df_train.iloc[train_idx].selected_text.values
        )
    
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=config.TRAIN_BATCH_SIZE,
            num_workers=0, 
            pin_memory=True
        )

        val_dataset = TweetDataset(
            tweet=df_train.iloc[val_idx].text.values,
            sentiment=df_train.iloc[val_idx].sentiment.values,
            selected_text=df_train.iloc[val_idx].selected_text.values
        )
    
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=config.VALID_BATCH_SIZE,
            num_workers=0, 
            pin_memory=True
        )
    
        del train_dataset, val_dataset
        gc.collect()


    with timer('create model'):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = TweetRoBERTaModel(config.ROBERTA_PATH)
        # model = TweetRoBERTaModelSimple(config.ROBERTA_PATH)
        # model = TweetRoBERTaModelConv1dHeadV2(config.ROBERTA_PATH)
        model = model.to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
        optimizer = transformers.AdamW(optimizer_parameters, lr=3e-5, correct_bias=False)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )
        
        # model = nn.DataParallel(model)
        
        # pretrain_path = 'models/exp11_fold0.pth'
        # model.load_state_dict(torch.load(pretrain_path))
        # LOGGER.info(f'pretrained model (exp11) loaded')


    with timer('training loop'):
        best_score = -999
        best_epoch = 0
        patience = 3
        p = 0
        for epoch in range(1, config.EPOCHS + 1):

            LOGGER.info("Starting {} epoch...".format(epoch))

            engine.train_fn(train_loader, model, optimizer, device, scheduler)
            score, val_loss = engine.eval_fn(val_loader, model, device)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(config.OUT_DIR, '{}_fold{}.pth'.format(EXP_ID, fold_id)))
                LOGGER.info("save model at score={} on epoch={}".format(best_score, best_epoch))
                p = 0
            
            if p > 0: 
                LOGGER.info(f'best score is not updated while {p} epochs of training')
            p += 1
            if p > patience:
                LOGGER.info(f'Early Stopping')
                break

        LOGGER.info("best score={} on epoch={}".format(best_score, best_epoch))


if __name__ == '__main__':

    fold0_only = config.FOLD0_ONLY

    for fold_id in range(config.NUM_FOLDS):

        LOGGER.info("#####")
        LOGGER.info("#####")
        LOGGER.info("Starting fold {} ...".format(fold_id))
        LOGGER.info("#####")
        LOGGER.info("#####")

        run_one_fold(fold_id)

        if fold0_only:
            LOGGER.info("This is fold0 only experiment.")
            break
