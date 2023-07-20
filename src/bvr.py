"""
Bert Variatinal Auto Encoder Recurrent Network
"""
import os

from copy import deepcopy

from dataclasses import dataclass
import typing

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

BERT_CONFIG_FIELDS = [
    "vocab_size",
    "type_vocab_size"
    "pad_token_id",
    "hidden_size",
    "hidden_dropout_prob",
    "max_position_embeddings",
    "layer_norm_eps",
    "position_embedding_type"
]

@dataclass
class BVRConfig:
    rnn_hidden_size: int
    rnn_num_layers: int
    rnn_dropout: float

    vae_latent_size: int

    bert_vocab_size: typing.Optional[int]=30
    bert_type_vocab_size: typing.Optional[int]=2
    bert_cls_token_id: typing.Optional[int]=2
    bert_pad_token_id: typing.Optional[int]=0
    bert_hidden_size: typing.Optional[int]=1024
    bert_hidden_dropout_prob: typing.Optional[float]=0.0
    bert_max_position_embeddings: typing.Optional[int]=40000
    bert_layer_norm_eps: typing.Optional[float]=1e-12
    bert_position_embedding_type: typing.Optional[str]="absolute"

    @property
    def rnn(self):
        return type('', (object,), {k[4:]: v for k, v in self.__dict__.items() if k[:3]=="rnn"})()

    @property
    def vae(self):
        return type('', (object,), {k[4:]: v for k, v in self.__dict__.items() if k[:3]=="vae"})()

    @property
    def bert(self):
        return type('', (object,), {k[5:]: v for k, v in self.__dict__.items() if k[:4]=="bert"})()
    
    @classmethod
    def from_config(cls, path, file_name="bvr_config.json"):
        if not os.path.isfile(os.path.join(path, file_name)):
            raise FileNotFoundError(f"Coudln't find {file_name} file in the specified directory: {path}")
        
        with open(os.path.join(path, file_name)) as f:
            config = json.load(f)

        return cls(**config)

    @classmethod
    def from_bert_config(cls, path, hidden_size, num_layers, dropout, latent_size, cls_token_id):
        bert_config = BertConfig.from_pretrained(path).to_dict()
        bert_config = {"bert_" + k: v for k, v in bert_config.items() if k in BERT_CONFIG_FIELDS}

        return BVRConfig(hidden_size, num_layers, dropout, latent_size, bert_cls_token_id=cls_token_id, **bert_config)
    
    def save(self, path, file_name="bvr_config.json"):
        with open(os.path.join(path, file_name), 'w', encoding='utf8') as f:
            f.write(json.dumps(self.__dict__, indent=4, skipkeys=True, sort_keys=False, separators=(',', ': '), ensure_ascii=False))

@dataclass
class BVREncoderOutput:
    hidden: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor

@dataclass
class BVRDecoderOutput:
    hidden: torch.Tensor
    logits: torch.Tensor

@dataclass
class BVROutput:
    encoder_hidden: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    decoder_hidden: torch.Tensor
    logits: torch.Tensor
    latent: torch.Tensor
    loss: typing.Optional[torch.Tensor]

class BVR(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embeddings = BertEmbeddings(config.bert)
        self.encoder = nn.GRU(
            input_size = config.bert_hidden_size,
            hidden_size = config.rnn_hidden_size,
            num_layers = config.rnn_num_layers,
            dropout = config.rnn_dropout,
            bidirectional = True,
        )
        self.decoder = nn.GRU(
            input_size = config.bert_hidden_size,
            hidden_size = config.rnn_hidden_size,
            num_layers = config.rnn_num_layers,
            dropout = config.rnn_dropout,
            bidirectional = False,
        )
        self.projection = nn.Linear(config.rnn_hidden_size, config.bert_vocab_size)

        self.hidden2mu = nn.Linear(config.rnn_hidden_size*2, config.vae_latent_size)
        self.hidden2logvar = nn.Linear(config.rnn_hidden_size*2, config.vae_latent_size)
        self.latent2embeddings = nn.Linear(config.vae_latent_size, config.bert_hidden_size)

    def encode(self, input):
        input = self.embeddings(input)
        _, hidden = self.encoder(input)
        hidden = torch.cat([hidden[-2], hidden[-1]], 1)

        return BVREncoderOutput(
            hidden = hidden,
            mu = self.hidden2mu(hidden),
            logvar = self.hidden2logvar(hidden)
        )
    
    def decode(self, input, latent, hidden=None):
        input = self.embeddings(input) + self.latent2embeddings(latent)
        output, hidden = self.decoder(input, hidden)
        logits = self.projection(output.view(-1, output.size(-1)))

        return BVRDecoderOutput(
            hidden = hidden,
            logits = logits.view(output.size(0), output.size(1), -1)
        )

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, input, target=None):
        encoder_output = self.encode(input)
        latent = self.reparametrize(encoder_output.mu, encoder_output.logvar)
        decoder_output = self.decode(input, latent)

        if target is not None:
            loss = F.cross_entropy(
                decoder_output.logits.view(-1, decoder_output.logits.size(-1)),
                target.flatten(),
                ignore_index=self.config.bert_pad_token_id,
                reduction="none"
            ).view(target.size()).mean()

        return BVROutput(
            encoder_hidden=encoder_output.hidden,
            mu = encoder_output.mu,
            logvar = encoder_output.logvar,
            decoder_hidden = decoder_output.hidden,
            logits = decoder_output.logits,
            latent = latent,
            loss = loss if target is not None else None
        )
    
    def generate(self, latent, max_len):
        sents = []
        input = torch.zeros(1, len(latent), dtype=torch.long, device=latent.device).fill_(self.config.bert_cls_token_id)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            decoder_output = self.decode(input, latent, hidden)
            input = torch.multinomial(
                decoder_output.logits.squeeze(dim=0).exp(),
                num_samples=1
            ).t()
            hidden = decoder_output.hidden
        return torch.cat(sents)

    @classmethod
    def from_pretrained_bert(cls, path, hidden_size, num_layers, dropout, latent_size, cls_token_id):
        config = BVRConfig.from_bert_config(path, hidden_size, num_layers, dropout, latent_size, cls_token_id)
        model = cls(config)
        model.embeddings = deepcopy(BertForMaskedLM.from_pretrained(path).bert.embeddings)
        return model

    @classmethod
    def from_pretrained(cls, path, weights_name="bvr_weights.bin", config_name="bvr_config.json"):
        config = BVRConfig.from_config(path, config_name)
        model = cls(config)
        model.load_state_dict(torch.load(os.path.join(path, weights_name)))

    def save(self, path, weights_name="bvr_weights.bin", config_name="bvr_config.json"):
        self.config.save(path, config_name)
        torch.save(self.state_dict(), os.path.join(path, weights_name))

class BVRTokenizer(BertTokenizer):
    def __call__(self, *args, **kwargs):
        tokens = super().__call__(*args, **kwargs)
        tokens.input_ids = tokens.input_ids.T
        tokens.token_type_ids = tokens.token_type_ids.T
        tokens.attention_mask = tokens.attention_mask.T
        return tokens