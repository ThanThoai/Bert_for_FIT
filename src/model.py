from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import sys
sys.path.append('..')


from .utils import cached_path
from .base import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

ACT2FN = {
    "gelu" : gelu,
    "relu" : torch.nn.functional.relu,
    "swish" : swish
}


class PreTrainedBertModel(nn.Module):
    
    def __init__(self, config, *input, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError("Error")
        self.config = config

    def __init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range) 
        elif isinstance(module, BertLayerNorm):
            module.beta.data.normal_(mean = 0.0, std = self.config.initializer_range)
            module.gamma.data.normal_(mean = 0.0, std = self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zeros_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, cache_dir = None, *inputs, **kwargs):
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        model = cls(config, *inputs, **kwargs)
        weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
        state_dict = torch.load(weights_path)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if tempdir:
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.__init_bert_weights)

    def forward(self, input_ids, token_type_ids = None, attention_mask = None, output_all_encoded_layers = True):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype = next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * (-10000.0)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers = output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output



class BertForFIT(PreTrainedBertModel):
    
    def __init__(self, config):
        super(BertForFIT, self).__init__(config)
        self.bert = BertModel(config)
        self.cls  = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)
        self.loss = nn.CrossEntropyLoss(reduction = 'none')
        self.vocab_size = self.bert.embeddings.word_embeddings.weight.size(0)

    
    def accuracy(self, output, target):
        output = torch.argmax(output, -1)
        return (output == target).float()

    def forward(self, input_ids, target):

        articles, articles_mask, ops, ops_mask, question_pos, mask = input_ids

        bsz = ops.size(0)
        opnum = ops.size(1)

        out, _ = self.bert(articles, attention_mask = articles_mask, output_all_encoded_layers = False)
        question_pos = question_pos.unsqueeze(-1)
        question_pos = question_pos.expand(bsz, opnum, out.size(-1))
        out = torch.gather(out, 1, question_pos)
        out = self.cls(out)

        out = out.view(bsz, opnum, 1, self.vocab_size)
        out = out.expand(bsz, opnum, 4, self.vocab_size)
        out = torch.gather(out, 3, ops)

        out *= ops_mask 
        out = out.sum(-1)
        out /= ops_mask.sum(-1)

        out = out.view(-1, 4)
        target = target.view(-1, )
        loss = self.loss(out, target)
        acc  = self.accuracy(out, target)

        loss = loss.view(bsz, opnum)
        acc = acc.view(bsz, opnum)

        loss *= mask
        acc *= mask

        acc = acc.sum(-1)

        acc = acc.sum()
        loss = loss.sum() / (mask.sum()) 
        return loss, acc

    def init_zero_weight(self, shape):
        weight = next(self.parameters())
        return weight.new_zeros(shape)