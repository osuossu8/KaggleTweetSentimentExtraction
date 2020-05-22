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

EXP_ID = "exp31"
import src.configs.config31 as config
import src.engine_with_aux_without_neutral_replacement as engine
from src.machine_learning_util import seed_everything, prepare_labels, DownSampler, timer, \
                                      to_pickle, unpickle
from src.model import TweetBERTBaseUncased, TweetModel, TweetModelLargeWWM, TweetRoBERTaModelV4
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


LOGGER_PATH = f"logs/log_{EXP_ID}.txt"
setup_logger(out_file=LOGGER_PATH)
LOGGER.info("seed={}".format(SEED))


def process_data_with_aux(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " " + " ".join(str(selected_text).split())

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    
    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets
    
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)
    
    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }
    
    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        
    if sentiment == 'neutral':
        sentiment_target = [1, 0]
    else:
        sentiment_target = [0, 1]
        
    
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets,
        'sentiment_target' : sentiment_target
    }


class TweetDatasetWithAux:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data_with_aux(
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
            'offsets': torch.tensor(data["offsets"], dtype=torch.long),
            'sentiment_target': torch.tensor(data["sentiment_target"], dtype=torch.float32)
        }


def run_one_fold(fold_id):

    with timer('load csv data'):

        debug = config.DEBUG
        df_train = pd.read_csv(config.TRAIN_PATH).dropna().reset_index(drop=True)

        if debug:
            df_train = df_train.sample(1000, random_state=SEED).dropna().reset_index(drop=True)

        # https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/142011
        # df_train.loc[df_train['sentiment']=='neutral', 'selected_text'] = df_train[df_train['sentiment']=='neutral']['text']

        num_folds = config.NUM_FOLDS
        kf = StratifiedKFold(n_splits = num_folds, random_state = SEED)
        splits = list(kf.split(X=df_train, y=df_train[['sentiment']]))
        train_idx = splits[fold_id][0]
        val_idx = splits[fold_id][1]

        print(len(train_idx), len(val_idx))

        gc.collect()


    with timer('prepare validation data'):
        train_dataset = TweetDatasetWithAux(
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

        val_dataset = TweetDatasetWithAux(
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

        model = TweetRoBERTaModelV4(config.ROBERTA_PATH)
        model = model.to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
        optimizer = transformers.AdamW(optimizer_parameters, lr=3e-5)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )

        model = nn.DataParallel(model)
        
        # pretrain_path = 'models/exp11_fold0.pth'
        # model.load_state_dict(torch.load(pretrain_path))
        # LOGGER.info(f'pretrained model (exp11) loaded')


    with timer('training loop'):
        best_score = -999
        min_loss = 999
        best_epoch = 0
        patience = 3
        p = 0
        for epoch in range(1, config.EPOCHS + 1):

            LOGGER.info("Starting {} epoch...".format(epoch))

            engine.train_fn(train_loader, model, optimizer, device, scheduler)
            score, val_loss = engine.eval_fn(val_loader, model, device)

            LOGGER.info(f"Jaccard Score = {score}")

            if val_loss < min_loss:
                min_loss = val_loss
                best_score = score
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(config.OUT_DIR, '{}_fold{}.pth'.format(EXP_ID, fold_id)))
                LOGGER.info("save model at score={} on epoch={}".format(best_score, best_epoch))
                p = 0
            
            if p > 0: 
                LOGGER.info(f'val loss is not updated while {p} epochs of training')
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
