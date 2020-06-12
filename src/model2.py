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


class TweetRoBERTaModelSimple(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelSimple, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.qa_outputs = nn.Linear(768*2, 2)
        torch.nn.init.normal_(self.qa_outputs.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        sequence_output, pooled_output, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        # out = out[-1]
        out = torch.cat((out[-1], out[-2]), dim=-1)

        # out = self.drop_out(out)
        # out = self.relu(self.l0(out))

        # logits = self.qa_outputs(sequence_output)
        logits = self.qa_outputs(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, dilation=1, dropout=0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, filter_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.model(x)


class TweetRoBERTaModelConv1d(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelConv1d, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(4584, 1024)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.l1 = nn.Linear(1024, 256)
        torch.nn.init.normal_(self.l1.weight, std=0.02)

        self.l2 = nn.Linear(256, 2)
        torch.nn.init.normal_(self.l2.weight, std=0.02)        

        self.relu = nn.ReLU(inplace=True)

        self.filters = [3, 5, 7]
        for filter_size in self.filters:
            setattr(
                self,
                f"seq{filter_size}", 
                nn.Sequential(
                    Conv1dBlock(128, 64, filter_size, dropout=0.1),
                    Conv1dBlock(64, 128, filter_size, dropout=0.1),
                ),
            )
            
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
                
    
    def forward(self, ids, mask, token_type_ids):
        _, _, hs = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        hs = torch.cat((hs[-1], hs[-2]), dim=-1)
        hs = self.drop_out(hs)
        
        outs = []
        for filter_size in self.filters:
            out = getattr(self, f"seq{filter_size}")(hs)
            outs.append(out)
            
        out = torch.cat(outs, axis=-1)
        
        out = self.relu(self.l0(out))
        out = self.relu(self.l1(out))
        logits = self.l2(out)

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


class TweetRoBERTaModelMK5(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelMK5, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)

        self.gru = nn.GRU(768 * 2, 256, bidirectional=True, batch_first=True)
        self.gru_attention = Attention(256 * 2, config.MAX_LEN)

        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.l1 = nn.Linear(256 * 2 + 768, 2)
        torch.nn.init.normal_(self.l1.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        seq_out, pooled_out, hs = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        hs = torch.cat((hs[-1], hs[-2]), dim=-1)
        hs = self.drop_out(hs)

        h_gru, _ = self.gru(hs)
        h_gru_attn = self.gru_attention(h_gru)

        logits = self.l0(hs)

        sentiment_cat = torch.cat([h_gru_attn, pooled_out], 1)
        sentiment_logits = self.l1(sentiment_cat)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits , sentiment_logits


class TweetRoBERTaModelMK6(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelMK6, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)

        self.gru = nn.GRU(768 * 2, 256, bidirectional=True, batch_first=True)
        self.gru_attention = Attention(256 * 2, config.MAX_LEN)

        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.l1 = nn.Linear(256 * 2, 3)
        torch.nn.init.normal_(self.l1.weight, std=0.02)

        self.l2 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l2.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        seq_out, pooled_out, hs = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        hs = torch.cat((hs[-1], hs[-2]), dim=-1)
        hs = self.drop_out(hs)

        h_gru, _ = self.gru(hs)
        h_gru_attn = self.gru_attention(h_gru)

        logits = self.l0(hs)

        sentiment_logits = self.l1(h_gru_attn)
        incorrect_logits = self.l2(pooled_out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits, sentiment_logits, incorrect_logits


class TweetRoBERTaModelMK7(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelMK7, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)

        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.l1 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l1.weight, std=0.02)

        self.l2 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l2.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        seq_out, pooled_out, hs = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        hs = torch.cat((hs[-1], hs[-2]), dim=-1)
        hs = self.drop_out(hs)

        logits = self.l0(hs)

        sentiment_logits = self.l1(pooled_out)
        incorrect_logits = self.l2(pooled_out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits, sentiment_logits, incorrect_logits

