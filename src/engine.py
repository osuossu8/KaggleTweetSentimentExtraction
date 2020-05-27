import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import src.configs.config as config


#def loss_fn(o1, o2, t1, t2):
#    l1 = nn.BCEWithLogitsLoss()(o1, t1)
#    l2 = nn.BCEWithLogitsLoss()(o2, t2)
#    return l1 + l2


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# not right imprementation for this comp
# https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/140942
# def jaccard(str1, str2):
#     a = str1.lower().split()
#     b = str2.lower().split()
#     c = set(a).intersection(set(b))
#     return float(len(c)) / (len(a) + len(b) - len(c))


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
        idx_end = idx_start
    
    filtered_output  = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    jaccards = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for bi, d in enumerate(tk0):

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
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
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
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
    
    print(f"Jaccard = {jaccards.avg}")
    return jaccards.avg


'''
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
        
        fin_outputs_start.append(torch.softmax(o1, 1).cpu().detach().numpy())
        fin_outputs_end.append(torch.softmax(o2, 1).cpu().detach().numpy())
        
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
      fin_padding_lens, fin_tweet_tokens, 
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
'''

