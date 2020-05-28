import torch
import torch.nn as nn
import transformers

import src.configs.config as config


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


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


class TweetRoBERTaModelMK2(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelMK2, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.5)
        self.l0 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        torch.nn.init.normal_(self.l0.bias, 0)


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
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


class TweetRoBERTaModelMK3(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelMK3, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out1 = nn.Dropout(0.5)
        self.drop_out2 = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 4, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        x = torch.stack([out[-1], out[-2], out[-3], out[-4]]) # 16, 128, 768
        x_mean = torch.mean(x, 0)
        x_max, _ = torch.max(x, 0)

        x = torch.cat([x_mean, x_max], -1) # 16, 128, 768 * 2
        x = self.drop_out1(x)

        x2 = torch.cat((out[-1], out[-2]), dim=-1) # 16, 128, 768 * 2
        x2 = self.drop_out2(x2)

        x = torch.cat([x, x2], -1)

        logits = self.l0(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


class TweetRoBERTaModelMK4(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelMK4, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)

        self.gru = nn.GRU(768 * 2, 256, bidirectional=True, batch_first=True)
        self.gru_attention = Attention(256 * 2, config.MAX_LEN)

        self.lstm = nn.LSTM(768 * 2, 256, bidirectional=True, batch_first=True)
        self.lstm_attention = Attention(256 * 2, config.MAX_LEN)

        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.l1 = nn.Linear(256 * 2, 2)
        torch.nn.init.normal_(self.l1.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)

        h_gru, _ = self.gru(out)
        h_gru_attn = self.gru_attention(h_gru)

        # h_lstm, _ = self.lstm(out)
        # hidden = torch.cat((h_gru, h_lstm), dim=-1)
        # hidden = h_gru + h_lstm

        # logits = self.l0(h_gru)
        logits = self.l0(out)

        sentiment_logits = self.l1(h_gru_attn)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits , sentiment_logits


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
        self.l0 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.l1 = nn.Linear(768, 3)
        torch.nn.init.normal_(self.l1.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        x = torch.stack([out[-1], out[-2], out[-3], out[-4]])
        x = torch.mean(x, 0)
        x = self.drop_out(x)

        sentiment_logit = torch.mean(x, 1)
        sentiment_logit = self.l1(sentiment_logit)

        logits = self.l0(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits, sentiment_logit


class TweetRoBERTaModelV7(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelV7, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.drop_out2 = nn.Dropout(0.5)
        self.l1 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l1.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        # out_logit = out[0] #16, 128, 768
        out_logit = torch.stack([out[-1], out[-2], out[-3], out[-4]])
        out_logit = torch.mean(out_logit, 0)

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)

        logits = self.l0(out)

        out_logit = self.drop_out2(out_logit)
        sentiment_logit = torch.mean(out_logit, 1) #16, 768
        sentiment_logit = self.l1(sentiment_logit)

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


class TweetBERTModelMK2(nn.Module):
    def __init__(self, bert_path):
        super(TweetBERTModelMK2, self).__init__()
        model_config = transformers.BertConfig.from_pretrained(bert_path)
        model_config.output_hidden_states = True 
        self.bert = transformers.BertModel.from_pretrained(bert_path, config=model_config)
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


class TweetALBERTModel(nn.Module):
    def __init__(self, albert_path):
        super(TweetALBERTModel, self).__init__()
        model_config = transformers.AlbertConfig.from_pretrained(albert_path)
        model_config.output_hidden_states = True
        self.bert = transformers.AlbertModel.from_pretrained(albert_path, config=model_config)
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
