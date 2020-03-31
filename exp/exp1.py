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
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from torchvision import models, transforms
from contextlib import contextmanager
from collections import OrderedDict
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import KFold, GroupKFold
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
import PIL
from PIL import Image

from tqdm import tqdm, tqdm_notebook, trange
import warnings
warnings.filterwarnings('ignore')
# from apex import amp

sys.path.append("/usr/src/app/kaggle/tweet-sentiment-extraction")

from src.machine_learning_util import seed_everything, prepare_labels, DownSampler, timer, \
                                      to_pickle, unpickle
from src.image_util import resize_to_square_PIL, pad_PIL, threshold_image, \
                           bbox, crop_resize, Resize, \
                           image_to_tensor, train_one_epoch, validate
from src.scheduler import GradualWarmupScheduler
# from src.layers import ResidualBlock, Mish, GeM


SEED = 1129
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

DIR_INPUT = 'inputs/'
OUT_DIR = 'models'

RESIZE = 128

# https://albumentations.readthedocs.io/en/latest/api/augmentations.html
data_transforms = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Cutout(p=0.5),
    A.Resize(RESIZE, RESIZE, p=1),
    A.Normalize(p=1.0),
    ToTensorV2(),
    ])

data_transforms_test = A.Compose([
    A.Resize(RESIZE, RESIZE, p=1),
    A.Normalize(p=1.0),
    ToTensorV2(),
    ])


class PlantDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, y=None, transform=None):
        self.df = df

        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        input_dic = {}
        row = self.df.iloc[idx]

        image_src = DIR_INPUT + '/images/' + row['image_id'] + '.jpg'
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        else:
            pass
        
        input_dic["image"] = image

        if self.y is not None:
           
            labels = np.array([row['healthy'], row['multiple_diseases'], row['rust'], row['scab']]).astype(np.float32)

            return input_dic, labels
        else:
            return input_dic


class PlantModel(nn.Module):
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.backbone = torchvision.models.resnet18(pretrained=True)
        
        in_features = self.backbone.fc.in_features

        self.logit = nn.Linear(in_features, num_classes)
        
    def forward(self, x):

        batch_size, C, H, W = x.shape
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        x = F.dropout(x, 0.25, self.training)

        x = self.logit(x)

        return x


def run_one_fold(fold_id, epochs, batch_size):

    with timer('load csv data'):
    
        train = pd.read_csv('inputs/train.csv')
        y = train[["healthy", "multiple_diseases", "rust", "scab"]]

        num_folds = 5
        kf = KFold(n_splits = num_folds, random_state = SEED)
        # kf = MultilabelStratifiedKFold(n_splits = num_folds, random_state = SEED)
        splits = list(kf.split(X=train, y=y))
        train_idx = splits[fold_id][0]
        val_idx = splits[fold_id][1]

        print(len(train_idx), len(val_idx))

        print(y.head())

        gc.collect()


    with timer('prepare validation data'):

        y_train = y.iloc[train_idx]
        train_dataset = PlantDataset(train.iloc[train_idx], y=y_train, transform=data_transforms)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size*4, shuffle=True, num_workers=0, pin_memory=True)
  
        y_val = y.iloc[val_idx]

        val_dataset = PlantDataset(train.iloc[val_idx], y=y_val, transform=data_transforms_test)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=True)

        del train_dataset, val_dataset
        gc.collect()


    with timer('create model'):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = PlantModel()
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        t_max=10
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=t_max)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1.1, total_epoch=5,
                                           after_scheduler=scheduler_cosine)

    with timer('training loop'):
        best_score = -999
        best_epoch = 0
        for epoch in range(1, epochs + 1):

            LOGGER.info("Starting {} epoch...".format(epoch))

            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

            LOGGER.info('Mean train loss: {}'.format(round(tr_loss, 5)))

            val_pred, y_true, val_loss = validate(model, val_loader, criterion, device)

            val_pred, le = prepare_labels(val_pred)
            
            score = 0
            for i in range(4):
                score += roc_auc_score(y_true[:, i], val_pred[:, i], average='macro')
            score = score/4

            LOGGER.info('Mean valid loss: {} score: {}'.format(round(val_loss, 5), round(score, 5)))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(OUT_DIR, '{}_fold{}.pth'.format(EXP_ID, fold_id)))
                to_pickle(os.path.join(OUT_DIR, "{}_fold{}_oof.pkl".format(EXP_ID, fold_id)), [val_idx, val_pred])
                LOGGER.info("save model at score={} on epoch={}".format(best_score, best_epoch))
            scheduler.step()

        LOGGER.info("best score={} on epoch={}".format(best_score, best_epoch))


if __name__ == '__main__':

    fold0_only = True

    for fold_id in range(5):

        LOGGER.info("Starting fold {} ...".format(fold_id))

        epochs = 10
        batch_size = 64

        run_one_fold(fold_id, epochs, batch_size)

        if fold0_only:
            LOGGER.info("This is fold0 only experiment.")
            break

