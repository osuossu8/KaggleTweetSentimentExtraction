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
        self.qa_outputs = nn.Linear(768, 2)
        torch.nn.init.normal_(self.qa_outputs.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        sequence_output, pooled_output, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        # out = out[-1]
        # out = torch.cat((out[-1], out[-2]), dim=-1)

        # out = self.drop_out(out)
        # out = self.relu(self.l0(out))

        logits = self.qa_outputs(sequence_output)
        # logits = self.qa_outputs(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


class TweetRoBERTaModelConv1dHead(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelConv1dHead, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.conv1d = nn.Conv1d(768, 1, 1)
        
        self.qa_outputs = nn.Linear(768, 2)
        torch.nn.init.normal_(self.qa_outputs.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        sequence_output, pooled_output, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        # out = out[0]

        x1 = self.drop_out(sequence_output)
        x1 = x1.transpose(2,1)
        x1 = self.conv1d(x1)
        start_logits = torch.squeeze(x1, 1)
        
        x2 = self.drop_out(sequence_output)
        x2 = x2.transpose(2,1)
        x2 = self.conv1d(x2)
        end_logits = torch.squeeze(x2, 1)

        return start_logits, end_logits


class TweetRoBERTaModelConv1dHeadV2(nn.Module):
    def __init__(self, roberta_path):
        super(TweetRoBERTaModelConv1dHeadV2, self).__init__()
        model_config = transformers.RobertaConfig.from_pretrained(roberta_path)
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.conv1d_1 = nn.Conv1d(768, 2, kernel_size = 2, stride = 1)
        # self.conv1d_2 = nn.Conv1d(64, 2, kernel_size = 2, stride = 1)
        
        self.leakey_relu = nn.LeakyReLU(inplace=True)
        self.dense = nn.Linear(96, 96)
        torch.nn.init.normal_(self.dense.weight, std=0.02)


    def forward(self, ids, mask, token_type_ids):
        sequence_output, pooled_output, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        x1 = self.drop_out(sequence_output)
        x1 = x1.transpose(2,1)
        x1 = self.conv1d_1(x1)
        x1 = torch.nn.functional.pad(x1, (0, 1), 'constant', 0)
        x1 = self.leakey_relu(x1)
        x1 = self.dense(x1)
        x1 = x1.transpose(2,1)
        
        start_logits, end_logits = x1.split(1, dim=-1)
        
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

