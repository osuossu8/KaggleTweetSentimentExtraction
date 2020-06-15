import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
from tqdm import tqdm

import src.configs.config51 as config


def remove_special_beginner(x):
    if x.startswith('_'):
        return ' '.join(x.split()[1:])
    #if x.startswith('-'):
    #    return ' '.join(x.split()[1:])
    #if x.startswith(':'):
    #    return ' '.join(x.split()[1:])
    #if x.startswith(';'):
    #    return ' '.join(x.split()[1:])
    return x

def remove_urls(x):
    x = str(x)
    x = re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "" ,x)
    return x

def remove_mention(x):
    x = str(x)
    return ' '.join([i for i in x.split() if '@' not in i])

def remove_hashtag(x):
    x = str(x)
    return ' '.join([i for i in x.split() if '#' not in i])

def remove_special(x):
    for i in ['[-O]', ' </3', '???????', '?????????', '????']:
        x = x.replace(i, '')
    return x


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    # loss_fct = nn.CrossEntropyLoss()
    loss_fct = nn.CrossEntropyLoss(ignore_index=start_logits.size(1))
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) * 0.5
    return total_loss


def KSLoss(preds, target):
    target = torch.nn.functional.one_hot(target, num_classes=preds.shape[1]).to('cuda', dtype=torch.float)
    pred_cdf = torch.cumsum(torch.softmax(preds, dim=1), dim=1)
    target_cdf = torch.cumsum(target, dim=1)
    error = (target_cdf - pred_cdf)**2
    return torch.mean(error)


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_jaccard_score(
    original_tweet, 
    target_string, 
    sentiment_val, 
    idx_start, 
    idx_end, 
    offsets,
    verbose=False):
    
    if idx_end < idx_start:
        x = original_tweet
        x = remove_special_beginner(x)
        x = remove_urls(x)
        x = remove_mention(x)
        x = remove_hashtag(x)
        x = remove_special(x)
        filtered_output = x

    filtered_output  = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    #if sentiment_val == "neutral":
    #    filtered_output = original_tweet

    if len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    jaccards = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for bi, d in enumerate(tk0):

        ids = d["ids"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"].to(device, dtype=torch.long)
        targets_end = d["targets_end"].to(device, dtype=torch.long)
        offsets = d["offsets"]

        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
        # loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end) \
        loss = (KSLoss(outputs_start, targets_start) + KSLoss(outputs_end, targets_end)) * 0.5
        loss.backward()
        optimizer.step()
        scheduler.step()

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            jaccard_scores.append(jaccard_score)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)


def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"].to(device, dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"].to(device, dtype=torch.long)
            targets_end = d["targets_end"].to(device, dtype=torch.long)
            offsets = d["offsets"]

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            #loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end) \
            loss = (KSLoss(outputs_start, targets_start) + KSLoss(outputs_end, targets_end)) * 0.5

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    
    return jaccards.avg, losses.avg


