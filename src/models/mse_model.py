#!/usr/bin/env python
__author__ = 'arenduchintala'
import math
import torch
import torch.nn as nn
import pdb

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from src.utils.utils import SPECIAL_TOKENS

from src.rewards import score_embeddings
from src.rewards import rank_score_embeddings
from src.opt.noam import NoamOpt


def batch_cosine_sim(a, b):
    eps = 1e-10
    dots = torch.matmul(a, b.transpose(0, 1))
    a_norm = torch.norm(a, 2, 1, keepdim=True)
    b_norm = torch.norm(b, 2, 1, keepdim=True)
    ab_norm = torch.matmul(a_norm, b_norm.transpose(0, 1))
    ab_norm[ab_norm <= 0.0] = eps
    return torch.div(dots, ab_norm)


class L2_MSE_CLOZE(nn.Module):
    def __init__(self,
                 encoder,
                 rnn,
                 linear,
                 l1_dict,
                 l2_encoder,
                 l2_dict,
                 l1_key,
                 l2_key,
                 iters,
                 loss_type):
        super().__init__()
        self.rnn = rnn
        self.rnn_size = self.rnn.hidden_size
        self.linear = linear
        self.encoder = encoder
        self.l2_encoder = l2_encoder
        self.tanh = torch.nn.Tanh()
        self.l1_dict = l1_dict
        self.l1_dict_idx = {v: k for k, v in l1_dict.items()}
        self.l2_dict = l2_dict
        self.l2_dict_idx = {v: k for k, v in l2_dict.items()}
        self.l1_key = l1_key
        self.l2_key = l2_key
        self.z = torch.zeros(1, 1, self.rnn_size, requires_grad=False)
        self.iters = iters
        self.loss_type = loss_type  # loss type used at training
        self.init_cuda()
        self.init_key()
        self.init_param_freeze()

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

    def mix_inputs(self, l1_channel, l2_channel, l1_idxs, l2_idxs):
        g_inp_ind = l2_idxs.unsqueeze(2).expand(l2_idxs.size(0), l2_idxs.size(1), l2_channel.size(2)).float()
        v_inp_ind = l1_idxs.unsqueeze(2).expand(l1_idxs.size(0), l1_idxs.size(1), l1_channel.size(2)).float()
        encoded = v_inp_ind * l1_channel + g_inp_ind * l2_channel
        return encoded

    def get_hiddens(self, inp, lengths, batch_size):
        packed_encoded = pack(inp, lengths, batch_first=True)
        # encoded = (batch_size x seq_len x embedding_size)
        packed_hidden, (h_t, c_t) = self.rnn(packed_encoded)
        hidden, lengths = unpack(packed_hidden, batch_first=True)
        z = self.z.expand(batch_size, 1, self.rnn_size)
        fwd_hidden = torch.cat((z, hidden[:, :-1, :self.rnn_size]), dim=1)
        bwd_hidden = torch.cat((hidden[:, 1:, self.rnn_size:], z), dim=1)
        # bwd_hidden = (batch_size x seq_len x rnn_size)
        # fwd_hidden = (batch_size x seq_len x rnn_size)
        hidden = torch.cat((fwd_hidden, bwd_hidden), dim=2)
        return hidden

    def get_acc(self, pred, ref, l1_data):
        cs = batch_cosine_sim(pred, ref)
        _, arg_top = torch.topk(cs, 1, 1)
        # for each pred get nearest neighbor
        arg_top = arg_top.squeeze(1)
        acc = float((arg_top == l1_data).nonzero().numel()) / float(l1_data.numel())
        assert 0.0 <= acc <= 1.0
        return acc

    def update_l2_encoder(self, out, l2_data, l2_idxs):
        l2_update_idxs = l2_data[l2_idxs == 1].view(-1)
        out_update = out[l2_idxs == 1, :].view(-1, out.shape[-1])
        idx2update = {}
        for l2_up, o_up in zip(l2_update_idxs, out_update):
            l2_up = l2_up.item()
            o_up = o_up.unsqueeze(0)
            if l2_up in idx2update:
                idx2update[l2_up] = torch.cat((idx2update[l2_up], o_up), dim=0)
            else:
                idx2update[l2_up] = o_up
        for i, up in idx2update.items():
            self.l2_encoder.weight.data[i] = up.mean(0)
        return True

    def forward(self, batch):
        lengths, l1_data, l2_data, ind, _ = batch
        l1_idxs = ind.eq(1).long()
        l2_idxs = ind.eq(2).long()
        for st in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.UNK]:  # SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.BOS]:
            if st in self.l1_dict:
                l1_idxs[l1_data.eq(self.l1_dict[st])] = 0
                l2_idxs[l1_data.eq(self.l1_dict[st])] = 0
        batch_size = l1_data.size(0)
        assert batch_size == 1
        l1_encoded = self.encoder(l1_data)
        l2_encoded = self.l2_encoder(l2_data)
        encoded = self.mix_inputs(l1_encoded, l2_encoded, l1_idxs, l2_idxs)
        encoded = self.tanh(encoded)
        continue_iter = True
        it = 0
        while continue_iter:
            it += 1
            e = encoded.sum().item()
            hidden = self.get_hiddens(encoded, lengths, batch_size)
            out = self.linear(hidden)
            encoded[l2_idxs == 1, :] = self.tanh(out[l2_idxs == 1, :])  # replace all l2_idxs with predictions...
            e2 = encoded.sum().item()
            continue_iter = e2 != e and it <= self.iters
        if l2_idxs.nonzero().numel() > 0:
            self.update_l2_encoder(out, l2_data, l2_idxs)
        return 0.

    def init_param_freeze(self,):
        for n, p in self.named_parameters():
            p.requires_grad = False
            print(n, p.requires_grad)
        return True

    def init_cuda(self,):
        self = self.cuda()
        self.z = self.z.cuda()
        return True

    def is_cuda(self,):
        return self.rnn.weight_hh_l0.is_cuda

    def get_weight(self,):
        #TODO: push this function into l2_encoder object
        if isinstance(self.l2_encoder, torch.nn.Embedding):
            weights = self.l2_encoder.weight.clone().detach()
        else:
            raise BaseException("unknown l2_encoder type")
        if self.is_cuda():
            weights = weights.cpu()
        return weights

    def update_g_weights(self, weights):
        #TODO: push this function into l2_encoder object
        if self.is_cuda():
            weights = weights.clone().cuda()
        else:
            weights = weights.clone()
        if isinstance(self.l2_encoder, torch.nn.Embedding):
            self.l2_encoder.weight.data = weights
        else:
            raise BaseException("unknown l2_encoder type")


########################################################################################################################
########################################################################################################################


class MSE_CLOZE(nn.Module):
    def __init__(self,
                 input_size,
                 rnn_size,
                 encoder,
                 l1_dict,
                 loss_type,
                 dropout=0.3,
                 max_grad_norm=5.):
        super().__init__()
        self.encoder = encoder
        #_d = self.encoder.weight.data
        #_d = _d.div(_d.norm(dim=-1, keepdim=True).expand_as(_d))
        #self.encoder.weight.data = _d
        for n, p in self.encoder.named_parameters():
            p.requires_grad = False
        self.dropout = torch.nn.Dropout(dropout)
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.l1_dict = l1_dict
        self.rnn = nn.LSTM(self.input_size, self.rnn_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)
        self.linear = nn.Linear(2 * self.rnn_size, self.input_size)
        self.tanh = torch.nn.Tanh()
        self.z = torch.zeros(1, 1, self.rnn_size, requires_grad=False)
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss = torch.nn.MSELoss(reduction='sum')
        elif self.loss_type == 'huber':
            self.loss = torch.nn.SmoothL1Loss(reduction='sum')
        elif self.loss_type == 'cs' or self.loss_type == 'cs_margin':
            self.loss = torch.nn.CosineEmbeddingLoss(reduction='sum')
        else:
            raise BaseException("unknown loss type")
        self.max_grad_norm = max_grad_norm
        self.init_cuda()
        self.init_param_freeze()
        self.init_optimizer('Adam')

    def init_cuda(self,):
        self = self.cuda()
        self.z = self.z.cuda()

    def init_param_freeze(self,):
        self.encoder.weight.requires_grad = False
        for n, p in self.named_parameters():
            print(n, p.requires_grad)

    def is_cuda(self,):
        return self.rnn.weight_hh_l0.is_cuda

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

    def get_hiddens(self, inp, lengths, batch_size):
        packed_encoded = pack(inp, lengths, batch_first=True)
        # encoded = (batch_size x seq_len x embedding_size)
        packed_hidden, (h_t, c_t) = self.rnn(packed_encoded)
        hidden, lengths = unpack(packed_hidden, batch_first=True)
        z = self.z.expand(batch_size, 1, self.rnn_size)
        fwd_hidden = torch.cat((z, hidden[:, :-1, :self.rnn_size]), dim=1)
        bwd_hidden = torch.cat((hidden[:, 1:, self.rnn_size:], z), dim=1)
        # bwd_hidden = (batch_size x seq_len x rnn_size)
        # fwd_hidden = (batch_size x seq_len x rnn_size)
        hidden = torch.cat((fwd_hidden, bwd_hidden), dim=2)
        return hidden

    def get_acc(self, pred, ref, l1_data):
        cs = batch_cosine_sim(pred, ref)
        _, arg_top = torch.topk(cs, 1, 1)
        # for each pred get nearest neighbor
        arg_top = arg_top.squeeze(1)
        acc = float((arg_top == l1_data).nonzero().numel()) / float(l1_data.numel())
        assert 0.0 <= acc <= 1.0
        return acc

    def get_loss(self, pred, target):
        if self.loss_type.startswith('cs'):
            loss = self.loss(pred, target, torch.ones(target.shape[0]).type_as(target))
            if self.loss_type.endswith('margin'):
                loss += self.loss(pred[1:, :], target[:-1, :],
                                  -1 * torch.ones(target.shape[0] - 1).type_as(target))
                loss += self.loss(pred[:-1, :], target[1:, :],
                                  -1 * torch.ones(target.shape[0] - 1).type_as(target))
        elif self.loss_type.startswith('mse') or self.loss_type.startswith('huber'):
            loss = self.loss(pred, target)
            if self.loss_type.endswith('margin'):
                raise BaseException("unknown loss type")
        else:
            raise BaseException("unknown loss type")

        return loss

    def forward(self, batch, get_acc=True):
        lengths, l1_data, _, ind, word_mask = batch
        l1_idxs = ind.eq(1).long()
        l2_idxs = ind.eq(2).long()
        for st in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.UNK]:  # SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.BOS]:
            if st in self.l1_dict:
                l1_idxs[l1_data.eq(self.l1_dict[st])] = 0
                l2_idxs[l1_data.eq(self.l1_dict[st])] = 0
        batch_size = l1_data.size(0)
        l1_encoded = self.tanh(self.encoder(l1_data))
        encoded = self.dropout(l1_encoded)
        rand = torch.zeros_like(encoded[word_mask == 1, :]).uniform_(-0.01, 0.01)  # replace with rand vector
        rand.requires_grad = False
        encoded[word_mask == 1, :] = rand
        hidden = self.get_hiddens(encoded, lengths, batch_size)
        hidden = self.dropout(hidden)
        out = self.tanh(self.linear(hidden))
        out = out[l1_idxs == 1, :]
        loss = self.get_loss(out, l1_encoded[l1_idxs == 1, :])
        if get_acc:
            acc = self.get_acc(out, self.encoder.weight.data, l1_data[l1_idxs == 1])
        else:
            acc = 0.0
        return loss, acc

    def do_backprop(self, batch):
        _l, _a = self(batch, False)
        _l.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()),
                                                   self.max_grad_norm)
        if math.isnan(grad_norm):
            print('skipping update grad_norm is nan!')
        else:
            self.optimizer.step()
        loss = _l.item()
        return loss, grad_norm, _a

    def save_model(self, path):
        torch.save(self, path)
