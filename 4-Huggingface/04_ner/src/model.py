# -*- coding: UTF-8 -*-
from torch import nn
from transformers import AutoModel

import params


class BertNerModel(nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.bert_ner = AutoModel.from_pretrained(params.MODEL_NAME)
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 768),
            nn.Dropout(dropout),
            nn.Linear(768, self.bert_ner.config.num_labels)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert_ner(input_ids, attention_mask=attention_mask)
        # (batch, max_length, 768)
        output = output['last_hidden_state']
        # (batch, max_length, 9)
        output = self.output_layer(output)
        # (batch, 9, max_length)
        output = output.transpose(1, 2)
        return output
