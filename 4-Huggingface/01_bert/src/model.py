# -*- coding: UTF-8 -*-
from torch import nn
from transformers import AutoModel

import params


class BertClassifier(nn.Module):

    def __init__(self, num_classes, frozen_layers=10):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(params.MODEL_NAME)
        self.linear = nn.Linear(768, num_classes)

        # freeze embedding and a part of encoder layer
        for name, param in self.bert.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = False
            elif 'encoder.layer' in name:
                layer = int(name.split('.')[2])
                if layer < frozen_layers:
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.linear(output.pooler_output)
        return output
