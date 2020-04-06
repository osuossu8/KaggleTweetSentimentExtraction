import numpy as np
import pandas as pd
import albumentations as A
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
from albumentations.pytorch import ToTensorV2
from torchvision import models, transforms
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

import src.config as config
import src.engine as engine
from src.machine_learning_util import seed_everything, prepare_labels, DownSampler, timer, \
                                      to_pickle, unpickle
from src.image_util import resize_to_square_PIL, pad_PIL, threshold_image, \
                           bbox, crop_resize, Resize, \
                           image_to_tensor, train_one_epoch, validate
from src.scheduler import GradualWarmupScheduler


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


EXP_ID = "exp1"
LOGGER_PATH = f"logs/log_{EXP_ID}.txt"
setup_logger(out_file=LOGGER_PATH)
LOGGER.info("seed={}".format(SEED))


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.l0 = nn.Linear(768, 2)
    
    def forward(self, ids, mask, token_type_ids):
        # not using sentiment at all
        sequence_output, pooled_output = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        # (batch_size, num_tokens, 768)
        logits = self.l0(sequence_output)
        # (batch_size, num_tokens, 2)
        # (batch_size, num_tokens, 1), (batch_size, num_tokens, 1)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # (batch_size, num_tokens), (batch_size, num_tokens)

        return start_logits, end_logits


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
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())
        
        len_st = len(selected_text)
        idx0 = -1
        idx1 = -1
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind: ind+len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1 + 1):
                if tweet[j] != " ":
                    char_targets[j] = 1
        
        tok_tweet = self.tokenizer.encode(sequence=self.sentiment[item], pair=tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[3:-1]
        # print(tok_tweet_tokens)
        # print(tok_tweet.offsets)
        # ['[CLS]', 'spent', 'the', 'entire', 'morning', 'in', 'a', 'meeting', 'w', '/', 
        # 'a', 'vendor', ',', 'and', 'my', 'boss', 'was', 'not', 'happy', 'w', '/', 'them', 
        # '.', 'lots', 'of', 'fun', '.', 'i', 'had', 'other', 'plans', 'for', 'my', 'morning', '[SEP]']
        targets = [0] * (len(tok_tweet_tokens) - 4)
        if self.sentiment[item] == "positive" or self.sentiment[item] == "negative":
            sub_minus = 8
        else:
            sub_minus = 7

        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1 - sub_minus:offset2 - sub_minus]) > 0:
                targets[j] = 1
        
        targets = [0] + [0] + [0] + targets + [0]

        # print(tweet)
        # print(selected_text)
        # print([x for i, x in enumerate(tok_tweet_tokens) if targets[i] == 1])
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1
        
        # print(targets_start)
        # print(targets_end)

        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * 3 + [1] * (len(tok_tweet_ids) - 3)

        padding_length = self.max_len - len(tok_tweet_ids)
        ids = tok_tweet_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        targets = targets + ([0] * padding_length)
        targets_start = targets_start + ([0] * padding_length)
        targets_end = targets_end + ([0] * padding_length)

        sentiment = [1, 0, 0]
        if self.sentiment[item] == "positive":
            sentiment = [0, 0, 1]
        if self.sentiment[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'tweet_tokens': " ".join(tok_tweet_tokens),
            'targets': torch.tensor(targets, dtype=torch.long),
            'targets_start': torch.tensor(targets_start, dtype=torch.long),
            'targets_end': torch.tensor(targets_end, dtype=torch.long),
            'padding_len': torch.tensor(padding_length, dtype=torch.long),
            'orig_tweet': self.tweet[item],
            'orig_selected': self.selected_text[item],
            'sentiment': torch.tensor(sentiment, dtype=torch.float),
            'orig_sentiment': self.sentiment[item]
        }


def run_one_fold(fold_id):

    with timer('load csv data'):

        DEBUG = True
        df_train = pd.read_csv(config.TRAIN_PATH)

        if DEBUG:
            df_train = df_train.sample(1000, random_state=SEED)

        num_folds = 5
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

        model = BERTBaseUncased()
        model = model.to(device)

        # criterion = nn.BCEWithLogitsLoss().to(device)

        # t_max=10
        # scheduler_cosine = CosineAnnealingLR(optimizer, T_max=t_max)
        # scheduler = GradualWarmupScheduler(optimizer, multiplier=1.1, total_epoch=5,
        #                                    after_scheduler=scheduler_cosine)

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
        optimizer = transformers.AdamW(optimizer_parameters, lr=5e-5)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )

        model = nn.DataParallel(model)


    with timer('training loop'):
        best_score = -999
        best_epoch = 0
        for epoch in range(1, config.EPOCHS + 1):

            LOGGER.info("Starting {} epoch...".format(epoch))

            engine.train_fn(train_loader, model, optimizer, device, scheduler)
            score, val_outputs = engine.eval_fn(val_loader, model, device)

            LOGGER.info(f"Jaccard Score = {score}")

            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(config.OUT_DIR, '{}_fold{}.pth'.format(EXP_ID, fold_id)))
                to_pickle(os.path.join(config.OUT_DIR, "{}_fold{}_oof.pkl".format(EXP_ID, fold_id)), [val_idx, val_outputs])
                LOGGER.info("save model at score={} on epoch={}".format(best_score, best_epoch))

        LOGGER.info("best score={} on epoch={}".format(best_score, best_epoch))


if __name__ == '__main__':

    fold0_only = True

    for fold_id in range(5):

        LOGGER.info("Starting fold {} ...".format(fold_id))

        run_one_fold(fold_id)

        if fold0_only:
            LOGGER.info("This is fold0 only experiment.")
            break

