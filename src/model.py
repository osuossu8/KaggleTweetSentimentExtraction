import torch
import torch.nn as nn
import transformers

import src.configs.config as config


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


class TweetRoBERTaModel(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModel, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


class TweetRoBERTaModelV2(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelV2, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.l0_s = nn.Linear(768, 1)
        self.l0_e = nn.Linear(768, 1)
        torch.nn.init.normal_(self.l0_s.weight, std=0.02)
        torch.nn.init.normal_(self.l0_e.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        
        out_start, out_end = out.split(768, dim=-1)

        start_logits = self.l0_s(out_start)
        end_logits = self.l0_e(out_end) 
        
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits


class TweetRoBERTaModelV3(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelV3, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
        self.l1 = nn.Linear(768 * 2, 3)
        torch.nn.init.normal_(self.l1.weight, std=0.02)
        
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        
        sentiment_logit = torch.mean(out, 1)
        sentiment_logit = self.l1(sentiment_logit)       
        
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits, sentiment_logit


class TweetRoBERTaModelV4(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelV4, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)

        logits = self.l0(out)

        sentiment_logit = torch.mean(logits, 1)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits, sentiment_logit


class TweetRoBERTaModelV5(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelV5, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.5)
        self.l0 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        x = torch.stack([out[-1], out[-2], out[-3], out[-4]])
        x = torch.mean(x, 0)
        x = self.drop_out(x)
        logits = self.l0(x)

        sentiment_logit = torch.mean(logits, 1)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits, sentiment_logit


class TweetRoBERTaModelV6(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelV6, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.l1 = nn.Linear(768 * 4, 768 * 2)
        torch.nn.init.normal_(self.l1.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)

        sentiment_logit_mean = torch.mean(out, 1)
        sentiment_logit_max, _ = torch.max(out, 1)
        sentiment_logit = torch.cat([sentiment_logit_mean, sentiment_logit_max], 1)
        sentiment_logit = self.l1(sentiment_logit)

        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits, sentiment_logit


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


class TweetModelLargeWWM(transformers.BertPreTrainedModel):
    def __init__(self, model_name, conf):
        super(TweetModelLargeWWM, self).__init__(conf)
        self.bert = transformers.BertModel.from_pretrained(model_name, output_hidden_states = True)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(1024 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


class TweetBERTBaseUncased(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetBERTBaseUncased, self).__init__(conf)
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config=conf)
        self.dropout = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 4, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        # not using sentiment at all
        sequence_output, pooled_output, hidden_states = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat([hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]], dim=-1)
        out = self.dropout(out)

        # (batch_size, num_tokens, 768)
        logits = self.l0(out)
        # (batch_size, num_tokens, 2)
        # (batch_size, num_tokens, 1), (batch_size, num_tokens, 1)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # (batch_size, num_tokens), (batch_size, num_tokens)

        return start_logits, end_logits
