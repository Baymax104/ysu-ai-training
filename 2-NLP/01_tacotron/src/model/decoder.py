# -*- coding: UTF-8 -*-
import torch
from torch import nn

from .attention import Attention
from .prenet import PreNet
from .stopnet import StopNet


class AttentionRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(AttentionRNN, self).__init__()
        self.gru = nn.GRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )

    def forward(self, x, hidden):
        return self.gru(x, hidden)


class DecoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )

    def forward(self, x, hidden):
        return self.gru(x, hidden)


class Decoder(nn.Module):
    r"""
    Shapes:
        - inputs: (batch, time_step, in_features)
        - outputs: (batch, frame_features, _)
    """

    def __init__(self, in_features, frame_features, r):
        super().__init__()
        self.in_features = in_features
        self.frame_features = frame_features
        self.r = r

        self.prenet = PreNet(frame_features, out_features=(256, 128))
        self.attention_rnn = AttentionRNN(input_size=128, hidden_size=256)
        self.attention = Attention(query_dim=256, embedding_dim=in_features, attention_dim=128)
        self.projection_to_decoder = nn.Linear(in_features=128, out_features=256)
        self.decoder_rnns = nn.ModuleList([DecoderRNN(input_size=256, hidden_size=256) for _ in range(2)])
        self.projection_to_mel = nn.Linear(in_features=256, out_features=frame_features * r)
        self.stopnet = StopNet(256 + frame_features * r)

        self.attention_rnn_hidden = None
        self.decoder_rnn_hiddens = None

    def __init_states(self, x):
        batch_size = x.size(dim=0)
        # (batch, 256)
        self.attention_rnn_hidden = torch.zeros(1, device=x.device).repeat(batch_size, 256)
        # (batch, 256)
        self.decoder_rnn_hiddens = [torch.zeros(1, device=x.device).repeat(batch_size, 256) for _ in
                                    range(len(self.decoder_rnns))]

    def decode(self, inputs, frame_input):
        # Prenet
        # (batch, 128)
        prenet_output = self.prenet(frame_input)
        # Attention RNN
        # (batch, time_step, 256)
        self.attention_rnn_hidden = self.attention_rnn(prenet_output, self.attention_rnn_hidden).clone()
        # Attention
        # (batch, 128)
        context_vector = self.attention(inputs, self.attention_rnn_hidden)
        # (batch, 256)
        decoder_input = self.projection_to_decoder(context_vector)

        # Decoder RNNs
        # (batch, 256)
        for i, decoder_rnn in enumerate(self.decoder_rnns):
            output = decoder_rnn(decoder_input, self.decoder_rnn_hiddens[i])
            self.decoder_rnn_hiddens[i] = output
            decoder_input = decoder_input + output  # Residual connection
        decoder_output = decoder_input

        # generate r frames
        # (batch, frame_features * r)
        output = self.projection_to_mel(decoder_output)

        # predict stop token
        stopnet_input = torch.cat([decoder_output, output], dim=-1)
        stop_token = self.stopnet(stopnet_input)
        return output, stop_token

    def forward(self, x):
        # (batch, time_step, 256)
        batch_size = x.size(dim=0)
        self.__init_states(x)
        frame_input = torch.zeros(1, device=x.device).repeat(batch_size, self.frame_features)
        outputs = []
        stop_tokens = []
        t = 0
        # for every time step
        while len(outputs) < x.size(dim=1):
            if t > 0:
                new_frame = outputs[t - 1]
                # use last frame
                frame_input = new_frame[:, self.frame_features * (self.r - 1):]
            output, stop_token = self.decode(x, frame_input)
            outputs.append(output)
            stop_tokens.append(stop_token)
            t += 1
        # (batch, time_step)
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1).squeeze()
        # (time_step, batch, frame_features * r)
        # (batch, time_step, frame_features * r)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        outputs = outputs.view(outputs.size(0), -1, self.frame_features)
        outputs = outputs.transpose(1, 2)
        return outputs, stop_tokens
