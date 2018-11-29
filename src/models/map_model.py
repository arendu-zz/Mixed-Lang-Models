#!/usr/bin/env python
__author__ = 'arenduchintala'
import math
import torch
import torch.nn as nn
import pdb

from src.models.model import VarLinear
from src.models.model import VarEmbedding
from src.models.model import VariationalEmbeddings

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from src.utils.utils import SPECIAL_TOKENS
from src.opt.noam import NoamOpt

from src.rewards import score_embeddings


class MapLinear(nn.Module):
    def __init__(self, l1_weights, map_weights):
        super(MapLinear, self).__init__()
        self.l1_weights = l1_weights
        self.l1_weights.requires_grad = False
        #shape = l1_vocab x emb_size
        #l1_voc, l2_voc = self.map.shape
        self.map_weights = map_weights
        self.map_weights.requires_grad = True
        self.register_buffer('l1_weight', self.l1_weights)
        self.register_buffer('map_weight', self.map_weights)

    def get_l2_weights(self,):
        l2_weights = torch.nn.functional.softmax(self.map_weights, dim=1)
        l2_weights = l2_weights.matmul(self.l1_weights)
        return l2_weights

    def forward(self, x):
        l2_weights = self.get_l2_weights()
        if x.dim() > 1:
            assert x.size(-1) == l2_weights.size(-1)
            return torch.matmul(x, l2_weights.transpose(0, 1))
        else:
            raise BaseException("x should be at least 2 dimensional")
        return None


class MapEmbedding(nn.Module):
    def __init__(self, l1_weights, map_weights):
        super(MapEmbedding, self).__init__()
        self.l1_weights = l1_weights
        self.l1_weights.requires_grad = False
        self.map_weights = map_weights
        self.map_weights.requires_grad = True
        self.register_buffer('l1_weight', self.l1_weights)
        self.register_buffer('map_weight', self.map_weights)

    def get_l2_weights(self,):
        l2_weights = torch.nn.functional.softmax(self.map_weights, dim=1)
        l2_weights = l2_weights.matmul(self.l1_weights)
        return l2_weights

    def get_map_param(self,):
        return self.map_weights

    def forward(self, x):
        l2_weights = self.get_l2_weights()
        embedding_dim = l2_weights.size(1)
        if x.dim() == 2:
            batch_size, seq_len = x.shape
            l2_emb = l2_weights[x.view(-1)]
            l2_emb = l2_emb.view(batch_size, seq_len, embedding_dim)
            return l2_emb
        elif x.dim() == 1:
            return l2_weights[x]


class CEncoderModelMap(nn.Module):
    L1_LEARNING = 'L1_LEARNING'  # updates only l1 params i.e. base language model
    L12_LEARNING = 'L12_LEARNING'  # updates both l1 params and l2 params (novel vocab embeddings)
    L2_LEARNING = 'L2_LEARNING'  # update only l2 params

    def __init__(self,
                 encoder,
                 decoder,
                 mode,
                 l1_dict,
                 l2_dict,
                 dropout=0.3,
                 max_grad_norm=5.,
                 size_average=False,
                 use_positional_embeddings=False):
        super().__init__()
        self.mode = mode
        self.l1_dict = l1_dict
        self.l2_dict = l2_dict
        self.encoder = encoder
        self.decoder = decoder
        self.dropout_val = dropout
        self.dropout = nn.Dropout(self.dropout_val)
        self.max_grad_norm = max_grad_norm
        self.emb_size = self.encoder.weight.shape[1]
        self.emb_max = self.encoder.weight.max().item()
        self.emb_min = self.encoder.weight.min().item()
        self.use_positional_embeddings = use_positional_embeddings
        self.l2_encoder = None #MapEmbedding(self.encoder.weight, map_weights)
        self.l2_decoder = None #MapLinear(self.encoder.weight, map_weights)

    def init_l2_weights(self, map_weights):
        self.map_param = torch.nn.Parameter(map_weights)
        self.l2_encoder = MapEmbedding(self.encoder.weight, self.map_param)
        self.l2_decoder = MapLinear(self.encoder.weight, self.map_param)
        if self.is_cuda():
            self.l2_encoder = self.l2_encoder.cuda()
            self.decoder = self.l2_decoder.cuda()
        return True

    def get_weight(self,):
        if self.is_cuda():
            weights = self.l2_encoder.map_weights.clone().detach().cpu()
        else:
            weights = self.l2_encoder.map_weights.clone().detach()
        return weights

    def forward(self, batcn):
        raise NotImplementedError

    def do_backprop(self, batch, l2_seen=None, total_batches=None):
        self.zero_grad()
        _l, _a = self(batch)
        if isinstance(self.encoder, VariationalEmbeddings):
            kl_loss = (1. / total_batches) * (self.encoder.log_q_w - self.encoder.log_p_w)
            _l += kl_loss

        _l.backward()

        if self.mode == CEncoderModelMap.L2_LEARNING:
            keep_grad = torch.zeros_like(self.l2_encoder.map_weights.grad)
            keep_grad[l2_seen, :] = 1.0
            self.l2_encoder.map_weights.grad *= keep_grad
            #for n, p in self.named_parameters():
            #    print(n, p.requires_grad, p.grad, p.grad.sum().item() if p.grad is not None else 'no sum')
            #    if p.grad is not None:
            #        pdb.set_trace()

        grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()),
                                                   self.max_grad_norm)
        if math.isnan(grad_norm):
            print('skipping update grad_norm is nan!')
        else:
            self.optimizer.step()
        loss = _l.item()
        #del _l
        #del batch
        return loss, grad_norm, _a

    def score_embs(self,):
        raise NotImplementedError("score embs not implemented")
        #l1_weights = self.encoder.weight.data
        #l2_weights = self.l2_encoder.weight.data

    def init_param_freeze(self, mode):
        self.mode = mode
        if self.mode == CEncoderModelMap.L12_LEARNING or self.mode == CEncoderModelMap.L2_LEARNING:
            self.dropout = nn.Dropout(0.0)  # we do this because cudnn RNN backward does not work in eval model...
            for p in self.parameters():
                p.requires_grad = False
            assert isinstance(self.l2_encoder, MapEmbedding)
            assert isinstance(self.l2_decoder, MapLinear)
            self.l2_encoder.map_weights.requires_grad = True
            self.l2_decoder.map_weights.requires_grad = True
            self.l2_encoder.l1_weights.requires_grad = False
            self.l2_encoder.l1_weights.requires_grad = False
            #print('L2_LEARNING, L1 Parameters frozen')
        elif self.mode == CEncoderModelMap.L1_LEARNING:
            self.dropout = nn.Dropout(self.dropout_val)
            for p in self.parameters():
                p.requires_grad = True
            #assert self.l2_encoder is None
            #assert self.l2_decoder is None
            #print('L1_LEARNING, L2 Parameters frozen')
            if isinstance(self.encoder, VarEmbedding):
                self.encoder.word_representer.set_extra_feat_learnable(False)
                assert isinstance(self.decoder, VarLinear)
                assert not self.decoder.word_representer.is_extra_feat_learnable

    def init_optimizer(self, type='Adam', lr=1.0):
        if type == 'Adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        elif type == 'SGD':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        elif type == 'noam':
            _optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
            self.optimizer = NoamOpt(self.emb_size, 100, _optimizer)
        else:
            raise NotImplementedError("unknown optimizer option")

    def set_key(self, l1_key, l2_key):
        self.l1_key = l1_key
        self.l2_key = l2_key

    def init_key(self,):
        if self.l1_key is not None:
            if self.is_cuda():
                self.l1_key = self.l1_key.cuda()
            else:
                pass
        if self.l2_key is not None:
            if self.is_cuda():
                self.l2_key = self.l2_key.cuda()
            else:
                pass

    def init_cuda(self,):
        raise NotImplementedError

    def is_cuda(self,):
        raise NotImplementedError

    def set_reset_weight(self,):
        self.reset_weight = self.l2_encoder.map_weight.detach().clone()

    def update_g_weights(self, map_weights):
        if self.is_cuda():
            map_weights = map_weights.clone().cuda()
        else:
            map_weights = map_weights.clone()
        self.l2_encoder.map_weight.data = map_weights
        self.l2_decoder.map_weight.data = map_weights

    def score_embeddings(self,):
        if isinstance(self.encoder, VarEmbedding):
            raise NotImplementedError("only word level scores")
        else:
            l1_embedding = self.encoder.weight.data
            l2_embedding = self.l2_encoder.get_l2_weights()
            s = score_embeddings(l2_embedding, l1_embedding, self.l2_key, self.l1_key)
            return s.item()

    def save_model(self, path):
        torch.save(self, path)


class CBiLSTMFastMap(CEncoderModelMap):

    def __init__(self, input_size, rnn_size, layers,
                 encoder, decoder,
                 mode,
                 l1_dict,
                 l2_dict,
                 size_average=False):
        super().__init__(encoder, decoder, mode, l1_dict, l2_dict)
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.rnn = nn.LSTM(self.input_size, self.rnn_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)
        #self.max_positional_embeddings = 100 #max_positional_embeddings
        #self.positional_embeddings = nn.Embedding(self.max_positional_embeddings, self.encoder.weight.shape[1])
        #self.layer_norm = nn.LayerNorm(input_size)

        self.linear = nn.Linear(2 * self.rnn_size, self.input_size)
        self.init_param_freeze(mode)
        self.loss = torch.nn.CrossEntropyLoss(size_average=size_average, reduce=True, ignore_index=0)
        #self.z = Variable(torch.zeros(1, 1, self.rnn_size), requires_grad=False)
        self.z = torch.zeros(1, 1, self.rnn_size, requires_grad=False)
        # .expand(batch_size, 1, self.rnn_size), requires_grad=False)
        self.l1_key = None
        self.l2_key = None
        self.init_optimizer(type='Adam')

    def init_cuda(self,):
        self = self.cuda()
        self.z = self.z.cuda()

    def is_cuda(self,):
        return self.rnn.weight_hh_l0.is_cuda

    def forward(self, batch):
        lengths, l1_data, l2_data, ind, word_mask = batch
        l1_idxs = ind.eq(1).long()
        l2_idxs = ind.eq(2).long()
        for st in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.UNK, SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.BOS]:
            if st in self.l1_dict:
                l1_idxs[l1_data.eq(self.l1_dict[st])] = 0
                l2_idxs[l1_data.eq(self.l1_dict[st])] = 0
        #if seen is not None:
        #    seen, seen_offset, seen_set = seen
        # if self.is_cuda():
        #     data = data.cuda()
        batch_size = l1_data.size(0)
        # max_seq_len = data.size(1)

        # l1_data = (batch_size x seq_len)
        # l2_data = (batch_size x seq_len)
        if self.mode == CEncoderModelMap.L2_LEARNING:
            l1_encoded = self.encoder(l1_data)
            l2_encoded = self.l2_encoder(l2_data)
            g_inp_ind = l2_idxs.unsqueeze(2).expand(l2_idxs.size(0), l2_idxs.size(1), l2_encoded.size(2)).float()
            v_inp_ind = l1_idxs.unsqueeze(2).expand(l1_idxs.size(0), l1_idxs.size(1), l1_encoded.size(2)).float()
            tmp_encoded = v_inp_ind * l1_encoded + g_inp_ind * l2_encoded
            encoded = l1_encoded * l1_idxs.unsqueeze(2).expand_as(l1_encoded).float() + \
                l2_encoded * l2_idxs.unsqueeze(2).expand_as(l2_encoded).float()
            assert (encoded - tmp_encoded).sum().item() == 0
            encoded = self.dropout(encoded)
        elif self.mode == CEncoderModelMap.L12_LEARNING:
            raise NotImplementedError("no longer supported")
            l1_encoded = self.encoder(l1_data)
            l2_encoded = self.l2_encoder(l2_data)
            g_inp_ind = l2_idxs.unsqueeze(2).expand(l2_idxs.size(0), l2_idxs.size(1), l2_encoded.size(2)).float()
            v_inp_ind = l1_idxs.unsqueeze(2).expand(l1_idxs.size(0), l1_idxs.size(1), l1_encoded.size(2)).float()
            encoded = v_inp_ind * l1_encoded + g_inp_ind * l2_encoded
            #encoded = l1_encoded * l1_idxs.unsqueeze(2).expand_as(l1_encoded).float() + \
            #    l2_encoded * l2_idxs.unsqueeze(2).expand_as(l2_encoded).float()
            #assert (encoded - tmp_encoded).sum().item() == 0
            encoded = self.dropout(encoded)
        else:
            #pos_data = torch.arange(lengths[0]).expand_as(l1_data).type_as(l1_data)
            #pos_data[pos_data > self.max_positional_embeddings - 1] = self.max_positional_embeddings - 1
            l1_encoded = self.encoder(l1_data)
            rand = torch.zeros_like(l1_encoded[word_mask == 1, :]).uniform_(self.emb_min, self.emb_max)
            rand.requires_grad = False
            l1_encoded[word_mask == 1, :] = rand
            #l1_pos_encoded = self.positional_embeddings(pos_data)
            #l1_encoded += l1_pos_encoded
            encoded = self.dropout(l1_encoded)

        packed_encoded = pack(encoded, lengths, batch_first=True)
        # encoded = (batch_size x seq_len x embedding_size)
        packed_hidden, (h_t, c_t) = self.rnn(packed_encoded)
        hidden, lengths = unpack(packed_hidden, batch_first=True)
        z = self.z.expand(batch_size, 1, self.rnn_size)
        fwd_hidden = torch.cat((z, hidden[:, :-1, :self.rnn_size]), dim=1)
        bwd_hidden = torch.cat((hidden[:, 1:, self.rnn_size:], z), dim=1)
        # bwd_hidden = (batch_size x seq_len x rnn_size)
        # fwd_hidden = (batch_size x seq_len x rnn_size)
        final_hidden = torch.cat((fwd_hidden, bwd_hidden), dim=2)
        final_hidden = self.dropout(final_hidden)
        final_hidden = self.linear(final_hidden)

        if self.mode == CEncoderModelMap.L1_LEARNING:
            l1_final_hidden = final_hidden[l1_idxs == 1, :]
            l1_out = self.decoder(l1_final_hidden)
            l1_pred = l1_out.argmax(1)
            acc = l1_pred.eq(l1_data[l1_idxs == 1]).sum().item()
            acc = float(acc) / float(l1_pred.size(0))
            loss = self.loss(l1_out, l1_data[l1_idxs == 1])
        elif self.mode == CEncoderModelMap.L2_LEARNING or self.mode == CEncoderModelMap.L12_LEARNING:
            l1_final_hidden = final_hidden[l1_idxs == 1, :]
            if l1_final_hidden.shape[0] > 0:
                l1_out = self.decoder(l1_final_hidden)
                l1_loss = self.loss(l1_out, l1_data[l1_idxs == 1])
            else:
                l1_loss = 0.
            l2_final_hidden = final_hidden[l2_idxs == 1, :]
            if l2_final_hidden.shape[0] > 0:
                l2_out = self.l2_decoder(l2_final_hidden)
                l2_loss = self.loss(l2_out, l2_data[l2_idxs == 1])
            else:
                l2_loss = 0.
            loss = l1_loss + l2_loss
            acc = None  # TODO
        else:
            raise BaseException("unknown learning type")
        return loss, acc
