# -*- coding: UTF-8 -*-
from torch import nn
from transformers import AutoModel

import params


class DistilBertModel(nn.Module):

    def __init__(self, dropout=0.3):
        super().__init__()
        self.distilbert = AutoModel.from_pretrained(params.MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # extract [CLS] representation
        # (batch, max_length, 768)
        output = output['last_hidden_state'][:, 0, :]
        output = self.dropout(output)
        output = self.classifier(output)
        return output
