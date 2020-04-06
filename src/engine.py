import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import src.config as config


def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return l1 + l2


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
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


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d, in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(o1, o2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


def eval_fn(data_loader, model, device):
    model.eval()
    fin_outputs_start = []
    fin_outputs_end = []
    fin_padding_lens = []
    fin_tweet_tokens = []
    fin_orig_sentiment = []
    fin_orig_selected = []
    fin_orig_tweet = []
    fin_tweet_token_ids = []
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for bi, d, in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        tweet_tokens = d['tweet_tokens']
        padding_len = d['padding_len']
        orig_sentiment = d['orig_sentiment']
        orig_selected = d['orig_selected']
        orig_tweet = d['orig_tweet']
        

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        fin_outputs_start.append(torch.sigmoid(o1).cpu().detach().numpy())
        fin_outputs_end.append(torch.sigmoid(o2).cpu().detach().numpy())
        
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
        fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())
        
        fin_tweet_tokens.extend(tweet_tokens)
        fin_orig_sentiment.extend(orig_sentiment)
        fin_orig_selected.extend(orig_selected)
        fin_orig_tweet.extend(orig_tweet)
        
    fin_outputs_start = np.vstack(fin_outputs_start)
    fin_outputs_end = np.vstack(fin_outputs_end)

    fin_tweet_token_ids = np.vstack(fin_tweet_token_ids)

    val_outputs = (
      fin_outputs_start, fin_outputs_end, fin_tweet_token_ids,
      fin_padding_lens, fin_tweet_tokens
      fin_orig_sentiment, fin_orig_selected, fin_orig_tweet
    )
    
    threshold = 0.2
    jaccards = []
    for j in range(len(fin_tweet_tokens)):
        target_string = fin_orig_selected[j]
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_lens[j]
        original_tweet = fin_orig_tweet[j]
        sentiment_val = fin_orig_sentiment[j]
        
        if padding_len > 0:
            mask_start = fin_outputs_start[j, 3:-1][:-padding_len] >= threshold
            mask_end = fin_outputs_end[j, 3:-1][:-padding_len] >= threshold
            tweet_token_ids = fin_tweet_token_ids[j, 3:-1][:-padding_len]
        else:
            mask_start = fin_outputs_start[j, 3:-1] >= threshold
            mask_end = fin_outputs_end[j, 3:-1] >= threshold
            tweet_token_ids = fin_tweet_token_ids[j, 3:-1]

            
        mask = [0] * len(mask_start)
        idx_start = np.nonzero(mask_start)[0]
        idx_end = np.nonzero(mask_end)[0]
        
        if len(idx_start) > 0:
            idx_start = idx_start[0]
            if len(idx_end) > 0:
                idx_end = idx_end[0]
            else:
                idx_end = idx_start
        else:
            idx_start = 0
            idx_end = 0
            
        for mj in range(idx_start, idx_end + 1):
            mask[mj] = 1
            
        # output_tokens = [x for p, x in enumerate(tweet_tokens.split()) if mask[p] == 1]
        # output_tokens = [x for x in output_tokens if x not in ('[CLS]', '[SEP]')]
        
        output_tokens = [x for p, x in enumerate(tweet_token_ids) if mask[p] == 1]
        # print(output_tokens)
        final_output = config.TOKENIZER.decode(output_tokens)
        # print(final_output)
        
        if sentiment_val == 'neutral' or len(original_tweet.split()) < 4:
            final_output = original_tweet
            
        jac = jaccard(target_string.strip().lower(), final_output.strip().lower())
        jaccards.append(jac)
    mean_jac = np.mean(jaccards)

    return mean_jac, val_outputs


