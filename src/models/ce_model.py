#!/usr/bin/env python
__author__ = 'arenduchintala'
import math
import torch
import torch.nn as nn
from collections import OrderedDict

from src.models.model_untils import BiRNNConextEncoder
from src.models.model_untils import SelfAttentionalContextEncoder

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from src.utils.utils import SPECIAL_TOKENS

from src.rewards import score_embeddings
from src.rewards import rank_score_embeddings
from src.rewards import get_nearest_neighbors

from src.opt.noam import NoamOpt
import pdb



def spell2mat(v2spell):
    spelling_mat = torch.Tensor(len(v2spell), len(v2spell[0])).fill_(0).long()
    for k, v in v2spell.items():
        spelling_mat[k] = torch.tensor(v)
    spelling_mat = spelling_mat[:, :-1]
    return spelling_mat


def make_l2_tied_encoder_decoder(l1_tied_enc_dec,
                                 v2i, l1_cgram2i_list, l1_cgram_spelling_mat_list,
                                 g2i, l2_cgram2i_list, l2_cgram_spelling_mat_list, l2_cgram_by_l1_spelling_mat_list,
                                 init_l2_prior, init_l2_with_l1,
                                 scale, init_range,
                                 l2_pos_weighted):
    if isinstance(l1_tied_enc_dec, CharTiedEncoderDecoder):
        c2i = l1_cgram2i_list[0]
        gc2i = l2_cgram2i_list[0]
        gv2spell_mat = l2_cgram_spelling_mat_list[0]
        l2_tied_enc_dec = CharTiedEncoderDecoder(char_vocab_size=len(gc2i),
                                                 char_embedding_size=l1_tied_enc_dec.char_embedding.embedding_dim,
                                                 word_vocab_size=len(g2i),
                                                 word_embedding_size=l1_tied_enc_dec.word_embedding_size,
                                                 spelling_mat=gv2spell_mat,
                                                 mode='l2',
                                                 pool=l1_tied_enc_dec.pool_type,
                                                 num_lang_bits=l1_tied_enc_dec.num_lang_bits)
        l2_tied_enc_dec.match_char_emb_params(c2i, gc2i, l1_tied_enc_dec.char_embedding)
        l2_tied_enc_dec.match_seq_params(l1_tied_enc_dec.seq)
        l2_tied_enc_dec.param_type = 'l2'
        return l2_tied_enc_dec
    elif isinstance(l1_tied_enc_dec, TiedEncoderDecoder):
        l2_tied_enc_dec = TiedEncoderDecoder(vocab_size=len(g2i),
                                             embedding_size=l1_tied_enc_dec.embedding.embedding_dim,
                                             mode='l2',
                                             vmat=None)
        l2_tied_enc_dec.embedding.weight.data.uniform_(-0.01, 0.01)
        l2_tied_enc_dec.param_type = 'l2'
        return l2_tied_enc_dec
    elif isinstance(l1_tied_enc_dec, CGramTiedEncoderDecoder):
        l2_cgram_vocab_sizes = [len(i) for i in l2_cgram2i_list]
        #l2_i2cgram_list = []
        #for i in l2_cgram2i_list:
        #    _i = {v: k for k, v in i.items()}
        #    l2_i2cgram_list.append(_i)
        l1_word_vocab_size = l1_tied_enc_dec.l1_word_vocab_size
        word_embedding_size = l1_tied_enc_dec.word_embedding_size
        l2_word_vocab_size = len(g2i)
        min_cgram = l1_tied_enc_dec.min_cgram
        max_cgram = l1_tied_enc_dec.max_cgram
        l2_cgram_vocab_sizes = l2_cgram_vocab_sizes[min_cgram:max_cgram]
        l2_cgram_spelling_mat_list = l2_cgram_spelling_mat_list[min_cgram:max_cgram]
        l2_cgram_by_l1_spelling_mat_list = l2_cgram_by_l1_spelling_mat_list[min_cgram:max_cgram]
        l2_tied_enc_dec = CGramTiedEncoderDecoder(l1_cgram_vocab_sizes=l1_tied_enc_dec.l1_cgram_vocab_sizes,
                                                  l1_cgram_spelling_mats=l1_tied_enc_dec.l1_cgram_spelling_mats,
                                                  l1_word_vocab_size=l1_word_vocab_size,
                                                  l2_cgram_vocab_sizes=l2_cgram_vocab_sizes,
                                                  l2_cgram_spelling_mats=l2_cgram_spelling_mat_list,
                                                  l2_cgram_by_l1_spelling_mats=l2_cgram_by_l1_spelling_mat_list,
                                                  l2_word_vocab_size=l2_word_vocab_size,
                                                  word_embedding_size=word_embedding_size,
                                                  mode='l2',
                                                  min_cgram=l1_tied_enc_dec.min_cgram,
                                                  max_cgram=l1_tied_enc_dec.max_cgram,
                                                  param_type='l2',
                                                  learn_main_l1_embs=False,
                                                  use_l2_subwords=init_l2_with_l1 == 'init_subwords',
                                                  init_range=init_range,
                                                  l2_pos_weighted=l2_pos_weighted)
        if init_l2_prior:
            l2_tied_enc_dec.init_l2_word_vecs_prior()
        if init_l2_with_l1 == 'init_main':
            l2_tied_enc_dec.copy_l1_params(l1_tied_enc_dec.l1_word_embedding, l1_tied_enc_dec.l1_params)
            l2_tied_enc_dec.init_with_l1_cgrams(scale)
        elif init_l2_with_l1 == 'init_subwords':
            l2_tied_enc_dec.copy_l1_subparams_into_l2_subparams(l1_cgram2i_list[min_cgram:max_cgram],
                                                                l1_tied_enc_dec.l1_params,
                                                                l2_cgram2i_list[min_cgram:max_cgram],
                                                                scale)
            l2_tied_enc_dec.copy_l1_wordparams_into_l2_wordparams(v2i, g2i,
                                                                  l1_tied_enc_dec.l1_word_embedding,
                                                                  scale)
        elif init_l2_with_l1 == 'no_init':
            l2_tied_enc_dec.l2_params[0].weight.data.uniform_(-init_range, init_range)
            pass
        return l2_tied_enc_dec
    else:
        raise NotImplementedError("unknown tied_encoder_decoder")


class TiedEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, mode, vmat=None):
        super().__init__()
        self.mode = mode
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.mode in ['l1', 'l2']
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 1.0)
        self.embedding.weight.data[0, :] = 0.
        if vmat is not None:
            self.embedding.weight.data = vmat
            self.pretrained = True
        else:
            self.pretrained = False
        self.decoder = torch.nn.Linear(embedding_size, vocab_size, bias=False)
        self.decoder.weight = self.embedding.weight
        self.param_type = ''

    def vocab_size(self,):
        return self.embedding.num_embeddings

    def init_param_freeze(self,):
        if self.mode == 'l1':
            if self.param_type == 'l1':
                self.embedding.weight.requres_grad = True and not self.pretrained
            else:
                raise NotImplementedError("weight type should only be l1 in mode=l1")
        else:
            assert not self.pretrained
            if self.param_type == 'l1':
                self.embedding.weight.requires_grad = False
            elif self.param_type == 'l2':
                self.embedding.weight.requires_grad = True
            else:
                raise NotImplementedError("unknown param_type/mode combination")
        return True

    def embedding_dim(self,):
        return self.embedding.embedding_dim

    def forward(self, data, mode):
        if mode == 'input':
            return self.embedding(data)
        elif mode == 'output':
            return self.decoder(data)
        else:
            raise NotImplementedError("unknown mode")

    def init_cuda(self,):
        self = self.cuda()
        return True

    def is_cuda(self,):
        return self.embedding.weight.data.is_cuda

    def get_word_vecs(self, on_device=True):
        if on_device:
            weights = self.embedding.weight.data.clone().detach()
        else:
            weights = self.embedding.weight.data.clone().detach().cpu()
        return weights

    def get_state_dict(self,):
        sd = OrderedDict()
        for k, v in self.embedding.state_dict().items():
            sd[k] = v.clone()
        return sd

    def set_state_dict(self, new_state_dict):
        self.embedding.load_state_dict(new_state_dict)
        self.decoder.weight = self.embedding.weight
        return True

    def get_undetached_state_dict(self,):
        return self.embedding.named_parameters()

    def get_weight_without_detach(self,):
        return self.embedding.weight

    def regularized_step(self, new_params):
        raise BaseException("don't regularize here, use l2 regularize instead")
        #self.l2_encoder.weight.data = 0.3 * l2_encoder_cpy + 0.7 * self.l2_encoder.weight.data
        self.set_params(0.3 * new_params + 0.7 * self.get_params())
        return True


class CGramTiedEncoderDecoder(nn.Module):
    def __init__(self,
                 l1_cgram_vocab_sizes, l1_cgram_spelling_mats, l1_word_vocab_size,
                 l2_cgram_vocab_sizes, l2_cgram_spelling_mats, l2_cgram_by_l1_spelling_mats, l2_word_vocab_size,
                 word_embedding_size,
                 mode, min_cgram, max_cgram, param_type, learn_main_l1_embs, use_l2_subwords,
                 init_range, l2_pos_weighted):
        super().__init__()

        self.dropout = nn.Dropout(0.2)

        self.l1_cgram_spelling_mats = l1_cgram_spelling_mats
        self.l1_cgram_vocab_sizes = l1_cgram_vocab_sizes
        self.l1_word_vocab_size = l1_word_vocab_size
        self.word_embedding_size = word_embedding_size
        self.learn_main_l1_embs = learn_main_l1_embs

        self.l2_cgram_vocab_sizes = l2_cgram_vocab_sizes
        self.l2_cgram_spelling_mats = l2_cgram_spelling_mats
        self.l2_cgram_by_l1_spelling_mats = l2_cgram_by_l1_spelling_mats
        self.l2_word_vocab_size = l2_word_vocab_size
        self.l2_pos_weighted = l2_pos_weighted

        self.mode = mode
        assert self.mode in ['l1', 'l2']
        self.param_type = param_type
        assert self.param_type in ['l1', 'l2']

        self.min_cgram = min_cgram # needed for l2_tied_enc_dec
        self.max_cgram = max_cgram

        self.l1_params = nn.ModuleList([])
        self.l1_cgram_spelling_embs = []

        self.l2_cgram_spelling_embs = []
        self.l2_cgram_by_l1_spelling_embs = []
        self.l2_params = nn.ModuleList([])

        for cg_sm, cg_vs in zip(self.l1_cgram_spelling_mats, self.l1_cgram_vocab_sizes):
            assert self.l1_word_vocab_size == cg_sm.shape[0], "spelling mat does not match word_vocab_size"
            s_emb = torch.nn.Embedding(cg_sm.shape[0], cg_sm.shape[1])
            s_emb.weight.data = cg_sm
            s_emb.weight.requires_grad = False
            self.l1_cgram_spelling_embs.append(s_emb)

            emb = torch.nn.Embedding(cg_vs, self.word_embedding_size, padding_idx=0)
            emb.weight.data.uniform_(-1.0, 1.0)
            emb.weight.data[0, :] = 0.
            self.l1_params.append(emb)

        self.l1_word_embedding = torch.nn.Embedding(l1_word_vocab_size, word_embedding_size, padding_idx=0)
        if self.learn_main_l1_embs:
            self.l1_word_embedding.weight.data.uniform_(-1.0, 1.0)
        else:
            self.l1_word_embedding.weight.data.fill_(0.0)
        self.l1_word_embedding.weight.data[0, :] = 0.
        self.l1_word_embedding.weight.requires_grad = self.learn_main_l1_embs

        self.use_cache_embedding = False
        self.use_l2_subwords = use_l2_subwords
        #self.l1_cached_word_embedding = None
        #self.l1_cached_word_embedding_decoder = None

        if self.mode == 'l2':
            self.l2_word_vecs_prior = None
            l2_word_embedding = torch.nn.Embedding(l2_word_vocab_size, word_embedding_size, padding_idx=0)
            print('using init_range', init_range)
            l2_word_embedding.weight.data.uniform_(-init_range, init_range)
            l2_word_embedding.weight.data[0, :] = 0.
            self.l2_params.append(l2_word_embedding)
            self.l1_subword_weight = 0.0 # init_range
            print(self.l1_subword_weight, 'is the l1_subword_weight')
            for cg_by_l1_sm in self.l2_cgram_by_l1_spelling_mats:
                assert self.l2_word_vocab_size == cg_by_l1_sm.shape[0], "spelling mat does not match word_vocab_size"
                s_by_l1_emb = torch.nn.Embedding(cg_by_l1_sm.shape[0], cg_by_l1_sm.shape[1])
                s_by_l1_emb.weight.data = cg_by_l1_sm
                s_by_l1_emb.weight.requires_grad = False
                self.l2_cgram_by_l1_spelling_embs.append(s_by_l1_emb)
            assert len(self.l2_cgram_by_l1_spelling_embs) == len(self.l1_params)

            print(self.use_l2_subwords, 'is the use_l2_subwords')
            for cg_sm, cg_vs in zip(self.l2_cgram_spelling_mats, self.l2_cgram_vocab_sizes):
                assert self.l2_word_vocab_size == cg_sm.shape[0], "spelling mat does not match word_vocab_size"
                s_emb = torch.nn.Embedding(cg_sm.shape[0], cg_sm.shape[1])
                s_emb.weight.data = cg_sm
                s_emb.weight.requires_grad = False
                if self.use_l2_subwords:
                    self.l2_cgram_spelling_embs.append(s_emb)

                emb = torch.nn.Embedding(cg_vs, self.word_embedding_size, padding_idx=0)
                emb.weight.data.uniform_(-init_range, init_range)
                emb.weight.data[0, :] = 0.
                if self.use_l2_subwords:
                    self.l2_params.append(emb)
        else:
            pass

    def copy_l1_wordparams_into_l2_wordparams(self, l1_v2idx, l2_v2idx, l1_word_embedding, scale):
        for v, idx in l2_v2idx.items():
            if l1_v2idx.get(v, float('inf')) < 50000:
                self.l2_params[0].weight.data[idx, :] = (l1_word_embedding.weight.data[l1_v2idx[v], :].clone() * scale)
        return True

    def copy_l1_subparams_into_l2_subparams(self, l1_cgram2i_list, l1_params, l2_cgram2i_list, scale):
        assert len(l1_cgram2i_list) == len(l1_params) == len(l2_cgram2i_list) == len(self.l2_params) - 1
        for l1_p, l1_c2i, l2_p, l2_c2i in zip(l1_params, l1_cgram2i_list,
                                              self.l2_params[1:], l2_cgram2i_list):
            for c, c_idx in l2_c2i.items():
                #print(c, l1_c2i.get(c, 0), 'idx in l1')
                if c in l1_c2i:
                    l2_p.weight.data[c_idx, :] = l1_p.weight.data[l1_c2i[c], :].clone() * scale
        return True

    def copy_l1_params(self, l1_word_embedding, l1_params):
        self.l1_word_embedding.weight.data = l1_word_embedding.weight.data
        assert len(self.l1_params) == len(l1_params)
        for self_l1_p, l1_p in zip(self.l1_params, l1_params):
            self_l1_p.weight.data = l1_p.weight.data.clone()
        return True

    def init_l2_word_vecs_prior(self,):
        v = torch.arange(self.l2_word_vocab_size).type_as(self.l2_params[0].weight.data).long()
        foo = self.summed_subword_emb(v, self.l1_params, self.l2_cgram_by_l1_spelling_embs, pos_weighted=True)
        self.l2_word_vecs_prior = foo.detach()
        return True

    def init_with_l1_cgrams(self, scale):
        v = torch.arange(self.l2_word_vocab_size).type_as(self.l2_params[0].weight.data).long()
        foo = self.summed_subword_emb(v, self.l1_params, self.l2_cgram_by_l1_spelling_embs, pos_weighted=False, scale=scale)
        self.l2_params[0].weight.data = foo.detach() #* (f / g)
        return True

    def embedding_dim(self,):
        return self.word_embedding_size

    def init_cache(self,):
        #assert self.mode == 'l2'
        #if self.param_type == 'l1':
        #    word_emb = self._compute_word_embeddings().detach()
        #    self.l1_cached_word_embedding = nn.Embedding(self.l1_word_vocab_size, self.word_embedding_size)
        #    self.l1_cached_word_embedding.weight.data = word_emb
        #    self.l1_cached_word_embedding_decoder = torch.nn.Linear(self.word_embedding_size,
        #                                                            self.l1_word_vocab_size,
        #                                                            bias=False)
        #    self.l1_cached_word_embedding_decoder.weight = self.l1_cached_word_embedding.weight
        #    self.use_cache_embedding = True
        #elif self.param_type == 'l2':
        #    pass
        #else:
        #    pass
        return True

    def summed_subword_emb(self, data, emb_list, spelling_emb_list, pos_weighted=False, scale=1.0, mean=True):
        #mean was false by default and only used for when param_type == 2
        assert len(emb_list) == len(spelling_emb_list)
        emb = 0
        data_dim = data.dim()
        for cg_emb, cg_spelling_emb in zip(emb_list, spelling_emb_list):
            cg_spelling = cg_spelling_emb(data).long()  # (data_dims, cgram_spelling_length)
            cg_spelling_lens = (cg_spelling != cg_emb.padding_idx).sum(data_dim)
            sub_emb = cg_emb(cg_spelling)  # (data_dims, cgram_spelling_length, word_embedding_size)
            if pos_weighted:
                _r = list(range(sub_emb.shape[data_dim] - 1, -1, -1))
                pos_wts = (torch.Tensor(_r).type_as(sub_emb) * (1.0 / (sub_emb.shape[data_dim] - 1))) ** 2
                pos_wts = pos_wts.unsqueeze(1).unsqueeze(0).expand_as(sub_emb)
                sub_emb = sub_emb * pos_wts
            sub_emb = sub_emb.sum(data_dim)  # (data_dims, word_embedding_Size)
            if mean:
                cg_spelling_lens = cg_spelling_lens.unsqueeze(data_dim).float().expand_as(sub_emb) + 1e-2
                sub_emb = sub_emb / cg_spelling_lens
            emb = emb + (sub_emb * scale)
        return emb

    def _compute_word_embeddings(self,):
        if self.param_type == 'l1':
            v = torch.arange(self.l1_word_vocab_size).type_as(self.l1_word_embedding.weight.data).long()
            word_emb = self.l1_word_embedding(v)
            word_emb = word_emb + self.summed_subword_emb(v, self.l1_params, self.l1_cgram_spelling_embs)
            if self.mode == 'l1':
                word_emb = self.dropout(word_emb)
            return word_emb
        elif self.param_type == 'l2':
            v = torch.arange(self.l2_word_vocab_size).type_as(self.l2_params[0].weight.data).long()
            word_emb = self.l2_params[0](v)
            if self.use_l2_subwords:
                #mean was false by default and only used for when param_type == 2
                word_emb = word_emb + self.summed_subword_emb(v, self.l2_params[1:],
                                                              self.l2_cgram_spelling_embs,
                                                              pos_weighted=self.l2_pos_weighted)
            return word_emb
        else:
            raise BaseException("bad param_type")

    def input_forward(self, data):
        # data shape = (bsz, seq)
        #if self.use_cache_embedding:
        #    assert self.param_type == 'l1'
        #    return self.l1_cached_word_embedding(data)

        if self.param_type == 'l1':
            word_emb = self.l1_word_embedding(data)  # (bsz, seq, word_embedding_size)
            word_emb = word_emb + self.summed_subword_emb(data, self.l1_params, self.l1_cgram_spelling_embs)
            if self.mode == 'l1':
                word_emb = self.dropout(word_emb)
            return word_emb
        elif self.param_type == 'l2':
            word_emb = self.l2_params[0](data)  # (bsz, seq, word_embedding_size)
            if self.use_l2_subwords:
                word_emb = word_emb + self.summed_subword_emb(data,
                                                              self.l2_params[1:],
                                                              self.l2_cgram_spelling_embs,
                                                              pos_weighted=self.l2_pos_weighted)
            #####word_emb = word_emb + self.l1_subword_weight * self.summed_subword_emb(data, self.l1_params, self.l2_cgram_by_l1_spelling_embs)
            return word_emb
        else:
            raise BaseException("bad param_type")

    def output_forward(self, data):
        #if self.use_cache_embedding:
        #    assert self.param_type == 'l1'
        #    return self.l1_cached_word_embedding_decoder(data)
        word_emb = self._compute_word_embeddings()
        return torch.matmul(data, word_emb.transpose(0, 1))

    def forward(self, data, mode):
        if mode == 'input':
            return self.input_forward(data)
        elif mode == 'output':
            return self.output_forward(data)
        else:
            raise NotImplementedError("unknown mode")

    def _disable_l1_param_learning(self,):
        assert self.mode == 'l2'
        self.l1_word_embedding.weight.requires_grad = False
        for p in self.l1_params:
            p.weight.requires_grad = False
        return True

    def init_param_freeze(self, ):
        if self.mode == 'l1':
            if self.param_type == 'l1':
                self.l1_word_embedding.weight.requires_grad = self.learn_main_l1_embs
                for p in self.l1_params:
                    p.weight.requires_grad = True
            if self.param_type == 'l2':
                raise BaseException("should not be here!")
        elif self.mode == 'l2':
            if self.param_type == 'l1':
                self.l1_word_embedding.weight.requires_grad = False
                for p in self.l1_params:
                    p.weight.requires_grad = False
                assert len(self.l2_params) == 0
            elif self.param_type == 'l2':
                self.l1_word_embedding.weight.requires_grad = False
                for p in self.l1_params:
                    p.weight.requires_grad = False
                for p in self.l2_params:
                    p.weight.requires_grad = True
            else:
                raise NotImplementedError("unknown param_type/mode combination")
        else:
            raise BaseException("unknown mode...")

        for sm_emb in self.l1_cgram_spelling_embs:
            sm_emb.weight.requires_grad = False

        for sm_emb in self.l2_cgram_spelling_embs:
            sm_emb.weight.requires_grad = False

        for sm_emb in self.l2_cgram_by_l1_spelling_embs:
            sm_emb.weight.requires_grad = False
        return True

    def init_cuda(self,):
        self = self.cuda()
        for sm_emb in self.l1_cgram_spelling_embs:
            sm_emb = sm_emb.cuda()
        for sm_emb in self.l2_cgram_spelling_embs:
            sm_emb = sm_emb.cuda()
        for sm_emb in self.l2_cgram_by_l1_spelling_embs:
            sm_emb = sm_emb.cuda()
        return True

    def is_cuda(self,):
        return self.l1_word_embedding.weight.data.is_cuda

    def get_word_vecs(self, on_device=True):
        weights = self._compute_word_embeddings().detach().clone()
        if on_device:
            pass
        else:
            weights = weights.cpu()
        return weights

    def get_undetached_state_dict(self,):
        assert self.param_type == 'l2'
        return self.l2_params.named_parameters()

    def get_state_dict(self,):
        assert self.param_type == 'l2'
        sd = OrderedDict()
        for k, v in self.l2_params.state_dict().items():
            sd[k] = v.clone()
        return sd

    def set_state_dict(self, new_state_dict):
        assert self.param_type == 'l2'
        self.l2_params.load_state_dict(new_state_dict)
        return True

    def vocab_size(self,):
        if self.param_type == 'l1':
            return self.l1_word_vocab_size
        elif self.param_type == 'l2':
            return self.l2_word_vocab_size
        else:
            raise BaseException("wrong mode set")


class CharTiedEncoderDecoder(nn.Module):
    def __init__(self, char_vocab_size, char_embedding_size, num_lang_bits,
                 word_vocab_size, word_embedding_size, spelling_mat, mode, pool):
        super().__init__()
        self.mode = mode
        self.pool_type = pool
        assert self.mode in ['l1', 'l2']
        self.word_vocab_size = word_vocab_size
        self.word_embedding_size = word_embedding_size
        self.char_embedding_size = char_embedding_size
        self.max_spelling_length = spelling_mat.shape[1]
        self.spelling_embedding = torch.nn.Embedding(word_vocab_size, self.max_spelling_length)
        self.spelling_embedding.weight.data = spelling_mat
        self.spelling_embedding.weight.requires_grad = False
        self.char_embedding = torch.nn.Embedding(char_vocab_size, char_embedding_size)
        self.num_lang_bits = num_lang_bits
        self.param_type = ''
        ##self.char_embedding.weight.data.uniform_(-0.01, 0.01)
        self.use_cache_embedding = False
        self.char_embedding.weight.data.normal_(0, 1.0)
        self.dropout = nn.Dropout(0.1)
        if self.pool_type.startswith('CNN'):
            ks = 4
            cnn = nn.Conv1d(self.char_embedding_size + self.num_lang_bits, self.word_embedding_size,
                            kernel_size=ks, padding=0, bias=False)
            # approximating many smaller kernel sizes with one large kernel by zeroing out kernel elements
            _k = self.word_embedding_size // ks
            for i in range(1, ks):
                x = torch.Tensor(_k, self.char_embedding_size + self.num_lang_bits, ks).uniform_(-0.05, 0.05)
                x[:, :, torch.arange(i, ks).long()] = 0
                cnn.weight.data[torch.arange((i - 1) * _k, i * _k).long(), :, :] = x
            if self.pool_type.startswith('CNNAvg'):
                mp = nn.AvgPool1d(2, self.max_spelling_length - (ks - 1))
            elif self.pool_type.startswith('CNNMax'):
                mp = nn.MaxPool1d(2, self.max_spelling_length - (ks - 1))
            elif self.pool_type.startswith('CNNLP'):
                mp = nn.LPPool1d(2, self.max_spelling_length - (ks - 1))
            else:
                raise BaseException("unknown pool_type")
            self.seq = nn.Sequential(cnn, mp)
        elif self.pool_type.startswith('RNN'):
            self.seq = nn.LSTM(input_size=self.char_embedding_size + self.num_lang_bits,
                               hidden_size=self.word_embedding_size,
                               num_layers=1,
                               batch_first=True)
        else:
            raise BaseException("unknown pool_type")

    def init_cache(self,):
        word_emb = self._compute_word_embeddings().detach()
        self.cached_word_embedding = nn.Embedding(self.word_vocab_size, self.word_embedding_size)
        self.cached_word_embedding.weight.data = word_emb
        self.cached_word_embedding_decoder = torch.nn.Linear(self.word_embedding_size,
                                                             self.word_vocab_size,
                                                             bias=False)
        self.cached_word_embedding_decoder.weight = self.cached_word_embedding.weight
        self.use_cache_embedding = True
        return True

    def match_char_emb_params(self, c2i, gc2i, char_embs):
        char_embs = char_embs.weight.clone().detach()
        for gc, gc_i in gc2i.items():
            ci = c2i.get(gc, c2i[SPECIAL_TOKENS.UNK_C])
            self.char_embedding.weight.data[gc_i, :] = char_embs[ci, :]
        return True

    def match_seq_params(self, seq_from_l1):
        self.seq.load_state_dict(seq_from_l1.state_dict())
        return True

    def embedding_dim(self,):
        return self.word_embedding_size

    def vocab_size(self,):
        return self.word_vocab_size

    def init_param_freeze(self, ):
        if self.mode == 'l1':
            if self.param_type == 'l1':
                for n, p in self.named_parameters():
                    if n != 'spelling_embedding.weight':
                        p.requires_grad = True
            else:
                raise NotImplementedError("weight type should only be l1 in mode=l1")
        else:
            if self.param_type == 'l1':
                for p in self.parameters():
                    p.requires_grad = False
            elif self.param_type == 'l2':
                for n, p in self.named_parameters():
                    if n != 'spelling_embedding.weight':
                        p.requires_grad = True
            else:
                raise NotImplementedError("unknown param_type/mode combination")
        self.spelling_embedding.weight.requires_grad = False
        return True

    def input_forward(self, data):
        #data shape = (bsz, seqlen)
        if self.use_cache_embedding:
            word_emb = self.cached_word_embedding(data)
        else:
            spelling = self.spelling_embedding(data).long()
            # spelling shape = (bsz, seqlen, max_spelling_length)
            char_emb = self.char_embedding(spelling)
            if self.mode == 'l1' and self.param_type == 'l1':
                char_emb = self.dropout(char_emb)
            bsz, seq_len, max_spelling_length, char_emb_size = char_emb.shape
            # char_emb shape = (bsz, seqlen, max_spelling_length, char_emb_size)
            char_emb = char_emb.view(-1, max_spelling_length, char_emb_size)
            # char_emb shape = (bsz * seqlen, max_spelling_length, char_emb_sze)
            if self.num_lang_bits > 0:
                lang_bits = torch.ones(1, 1, 1).type_as(char_emb).expand(char_emb.shape[0],
                                                                         char_emb.shape[1],
                                                                         self.num_lang_bits)
                lang_bits = lang_bits * 1 if self.param_type == 'l2' else lang_bits * 0
                char_emb = torch.cat([char_emb, lang_bits], dim=2)
            else:
                pass
            # char_emb shape = (bsz * seq_len, char_emb_size + 1, max_spelling_length)
            if self.pool_type.startswith('RNN'):
                _, (hn, cn) = self.seq(char_emb)
                word_emb = hn.squeeze(0) #out[:, -1, :]
                #assert (word_emb - out[:, -1, :]).sum().item() == 0
                #word_emb shape = (bsz * seq_len, word_emb_size)
            else:
                char_emb = char_emb.transpose(1, 2)
                word_emb = self.seq(char_emb).squeeze(2)
                #word_emb shape = (bsz * seq_len, word_emb_size)
            word_emb = word_emb.view(bsz, seq_len, -1)
            #if self.mode == 'l1' and self.param_type == 'l1':
            if self.mode == 'l1' and self.param_type == 'l1':
                word_emb = self.dropout(word_emb)
        return word_emb

    def _compute_word_embeddings(self,):
        spelling = self.spelling_embedding.weight.data.clone().detach().long()
        # spelling shape = (v, max_spelling_length)
        char_emb = self.char_embedding(spelling)
        if self.mode == 'l1' and self.param_type == 'l1':
            char_emb = self.dropout(char_emb)
        #v, max_spelling_length, char_emb_size = char_emb.shape
        if self.num_lang_bits > 0:
            lang_bits = torch.ones(1, 1, 1).type_as(char_emb).expand(char_emb.shape[0],
                                                                     char_emb.shape[1],
                                                                     self.num_lang_bits)
            lang_bits = lang_bits * 1 if self.param_type == 'l2' else lang_bits * 0
            char_emb = torch.cat([char_emb, lang_bits], dim=2)
        else:
            pass

        if self.pool_type.startswith('RNN'):
            _, (hn, cn) = self.seq(char_emb)
            word_emb = hn.squeeze(0)
        else:
            char_emb = char_emb.transpose(1, 2)
            word_emb = self.seq(char_emb).squeeze(2)
        if self.mode == 'l1' and self.param_type == 'l1':
            word_emb = self.dropout(word_emb)
        return word_emb

    def output_forward(self, data):
        if self.use_cache_embedding:
            return self.cached_word_embedding_decoder(data)
        else:
            word_emb = self._compute_word_embeddings()
            return torch.matmul(data, word_emb.transpose(0, 1))

    def forward(self, data, mode):
        if mode == 'input':
            return self.input_forward(data)
        elif mode == 'output':
            return self.output_forward(data)
        else:
            raise NotImplementedError("unknown mode")

    def init_cuda(self,):
        self = self.cuda()
        return True

    def is_cuda(self,):
        return self.char_embedding.weight.data.is_cuda

    def get_word_vecs(self, on_device=True):
        weights = self._compute_word_embeddings().detach().clone()
        if on_device:
            pass
        else:
            weights = weights.cpu()
        return weights

    def get_undetached_state_dict(self,):
        return self.named_parameters()

    def get_state_dict(self,):
        sd = OrderedDict()
        for k, v in self.state_dict().items():
            sd[k] = v.clone()
        return sd

    def set_state_dict(self, new_state_dict):
        self.load_state_dict(new_state_dict)
        return True

    def regularized_step(self, new_params):
        raise BaseException("don't regularize here, use l2 regularize instead")
        self.set_params(0.9 * new_params + 0.1 * self.get_params())
        return True


class ContextEncoder(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, encoded, lengths, forward_mode):
        pass

    def init_cuda(self):
        pass

    def is_cuda(self,):
        pass


class LMContextEncoder(ContextEncoder):
    def __init__(self, input_size, rnn_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.rnn_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=False)
        self.hidden_size = self.rnn_size
        self.projection = torch.nn.Linear(self.hidden_size, self.input_size)
        self.dropout = torch.nn.Dropout(0.1)

    def init_cuda(self,):
        self = self.cuda()
        return True

    def is_cuda(self,):
        for p in self.lstm.parameters():
            return p.is_cuda
        return False

    def forward(self, encoded, lengths, forward_mode):
        assert forward_mode in ['L1', 'L2']
        if forward_mode == 'L1':
            encoded = self.dropout(encoded)
        else:
            pass
        packed_encoded = pack(encoded, lengths, batch_first=True)
        packed_hidden, (h_t, c_t) = self.lstm(packed_encoded)
        hidden, lengths = unpack(packed_hidden, batch_first=True)
        z = torch.ones(hidden.size(0), 1, hidden.size(2)).type_as(hidden)
        z.requires_grad = False
        hidden = hidden[:, :-1, :]
        hidden = torch.cat([z, hidden], dim=1)
        if forward_mode == 'L1':
            hidden = self.dropout(hidden)
        else:
            pass
        projected_hidden = self.projection(hidden)
        return projected_hidden


class ClozeContextEncoder(ContextEncoder):
    def __init__(self, input_size, rnn_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.rnn_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.z = torch.zeros(1, 1, self.rnn_size, requires_grad=False)
        self.hidden_size = 2 * self.rnn_size
        self.projection = torch.nn.Linear(self.hidden_size, self.input_size)
        self.dropout = torch.nn.Dropout(0.1)

    def init_cuda(self,):
        self = self.cuda()
        self.z = self.z.cuda()
        return True

    def is_cuda(self,):
        for p in self.lstm.parameters():
            return p.is_cuda
        return False

    def forward(self, encoded, lengths, forward_mode):
        assert forward_mode in ['L1', 'L2']
        batch_size, seq_len, emb_size = encoded.shape
        if forward_mode == 'L1':
            encoded = self.dropout(encoded)
        else:
            pass
        packed_encoded = pack(encoded, lengths, batch_first=True)
        # encoded = (batch_size x seq_len x embedding_size)
        packed_hidden, (h_t, c_t) = self.lstm(packed_encoded)
        hidden, lengths = unpack(packed_hidden, batch_first=True)
        z = self.z.expand(batch_size, 1, self.rnn_size)
        fwd_hidden = torch.cat((z, hidden[:, :-1, :self.rnn_size]), dim=1)
        bwd_hidden = torch.cat((hidden[:, 1:, self.rnn_size:], z), dim=1)
        # bwd_hidden = (batch_size x seq_len x rnn_size)
        # fwd_hidden = (batch_size x seq_len x rnn_size)
        hidden = torch.cat((fwd_hidden, bwd_hidden), dim=2)
        if forward_mode == 'L1':
            hidden = self.dropout(hidden)
        else:
            pass
        projected_hidden = self.projection(hidden)
        return projected_hidden


class ClozeMaskContextEncoder(ContextEncoder):
    def __init__(self, input_size, rnn_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.rnn_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.hidden_size = 2 * self.rnn_size
        self.projection = torch.nn.Linear(self.hidden_size, self.input_size)
        self.mask_prob = 0.3
        noise_probs = torch.tensor([1.0 - self.mask_prob, self.mask_prob])
        self.noise_mask = torch.distributions.Categorical(probs=noise_probs)

    def init_cuda(self,):
        self = self.cuda()
        noise_probs = torch.tensor([1.0 - self.mask_prob, self.mask_prob]).cuda()
        self.noise_mask = torch.distributions.Categorical(probs=noise_probs)
        return True

    def is_cuda(self,):
        for p in self.lstm.parameters():
            return p.is_cuda
        return False

    def word_mask(self, encoded):
        n_idx = self.noise_mask.sample(sample_shape=(encoded.size(0), encoded.size(1)))
        encoded[n_idx == 1] = 0.
        return encoded

    def forward(self, encoded, lengths, forward_mode):
        assert forward_mode in ['L1', 'L2']
        if forward_mode == 'L1':
            encoded = self.word_mask(encoded)
        else:
            pass
        packed_encoded = pack(encoded, lengths, batch_first=True)
        packed_hidden, (h_t, c_t) = self.lstm(packed_encoded)
        hidden, lengths = unpack(packed_hidden, batch_first=True)
        if forward_mode == 'L1':
            projected_hidden = self.projection(torch.nn.functional.dropout(hidden, p=0.1, training=self.training))
        else:
            projected_hidden = self.projection(hidden)
        return projected_hidden


class CE_CLOZE(nn.Module):
    def __init__(self,
                 tied_encoder_decoder,
                 context_encoder,
                 l1_dict,
                 loss_at='all',
                 dropout=0.1,
                 max_grad_norm=5.):
        super().__init__()
        self.tied_encoder_decoder = tied_encoder_decoder
        self.context_encoder = context_encoder
        self.loss_at = loss_at
        self.l1_dict = l1_dict
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
        self.max_grad_norm = max_grad_norm
        #self.init_cuda()
        self.init_param_freeze()
        self.init_optimizer('Adam')

    def init_cuda(self,):
        self.context_encoder.init_cuda()
        self.tied_encoder_decoder.init_cuda()
        self = self.cuda()
        return True

    def init_param_freeze(self,):
        for n, p in self.context_encoder.named_parameters():
            p.requires_grad = True
        self.tied_encoder_decoder.init_param_freeze()
        for n, p in self.named_parameters():
            print(n, p.requires_grad)
        return True

    def is_cuda(self,):
        return self.context_encoder.is_cuda()

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

    #def get_noise_channel(self, l1_data, l1_encoded):
    #    n_idx = self.noise_mask.sample(sample_shape=(l1_data.size(0), l1_data.size(1)))
    #    n_idx[l1_data.eq(self.l1_dict[SPECIAL_TOKENS.PAD])] = 0
    #    n_idx[l1_data.eq(self.l1_dict[SPECIAL_TOKENS.EOS])] = 0
    #    n_idx[l1_data.eq(self.l1_dict[SPECIAL_TOKENS.BOS])] = 0
    #    nns = self.nn_mapper(l1_data)
    #    nns_idx = torch.empty_like(l1_data).random_(0, self.nn_mapper.embedding_dim - 1)
    #    nns_idx = nns_idx.unsqueeze(2).expand_as(nns)
    #    l1_nn = torch.gather(nns, 2, nns_idx)[:, :, 0]
    #    l1_nn_encoded = self.encoder(l1_nn)
    #    if self.noise_profile == 1:
    #        l1_noisy = torch.zeros_like(l1_encoded).type_as(l1_encoded)
    #        l1_noisy[n_idx == 0] = l1_encoded[n_idx == 0]                   # n_idx == 0 no noise
    #        l1_noisy[n_idx == 1] = 0.0                                      # n_idx == 1 blank vector
    #        l1_noisy[n_idx == 2] = l1_nn_encoded[n_idx == 2]                # n_idx == 2 nearest neighbors
    #    elif self.noise_profile == 2:
    #        l1_noisy = torch.zeros_like(l1_encoded).type_as(l1_encoded)
    #        l1_noisy[n_idx == 0] = l1_encoded[n_idx == 0]                   # n_idx == 0 no noise
    #        l1_noisy[n_idx == 1] = 0.0                                      # n_idx == 1 blank vector
    #        l1_noisy[n_idx == 2] = 0.0                                      # n_idx == 2 nearest neighbors
    #    elif self.noise_profile == 3:
    #        l1_rand = torch.empty_like(l1_data).random_(0, self.encoder.num_embeddings)
    #        l1_rand_encoded = self.encoder(l1_rand)
    #        l1_noisy = torch.zeros_like(l1_encoded).type_as(l1_encoded)
    #        l1_noisy[n_idx == 0] = l1_encoded[n_idx == 0]                   # n_idx == 0 no noise
    #        l1_noisy[n_idx == 1] = l1_nn_encoded[n_idx == 1]                # n_idx == 1 nearest neighbors
    #        l1_noisy[n_idx == 2] = l1_rand_encoded[n_idx == 2]              # n_idx == 2 rand words
    #    else:
    #        raise BaseException("unknown noise profile")

    #    n_idx[n_idx.ne(0)] = 1
    #    return l1_noisy, n_idx

    def get_acc(self, arg_top, l1_data):
        acc = float((arg_top == l1_data).nonzero().numel()) / float(l1_data.numel())
        assert 0.0 <= acc <= 1.0
        return acc

    def get_loss(self, pred, target):
        loss = self.loss(pred, target)
        return loss

    def forward(self, batch, get_acc=True):
        lengths, l1_data, _, ind = batch
        l1_idxs = ind.eq(1).long()
        l2_idxs = ind.eq(2).long()
        for st in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.UNK]:  # SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.BOS]:
            if st in self.l1_dict:
                l1_idxs[l1_data.eq(self.l1_dict[st])] = 0
                l2_idxs[l1_data.eq(self.l1_dict[st])] = 0
                ind[l1_data.eq(self.l1_dict[st])] = 0
        l1_encoded = self.tied_encoder_decoder(l1_data, mode='input')
        hidden = self.context_encoder(l1_encoded, lengths, forward_mode='L1')
        out = self.tied_encoder_decoder(hidden, mode='output')
        if self.loss_at == 'all':
            out_l = out.view(-1, out.size(-1))
            l1_data_l = l1_data.view(-1)
        else:
            raise BaseException("unknown loss at")
        loss = self.get_loss(out_l, l1_data_l)
        if get_acc:
            _, argmax = out_l.max(dim=1)
            acc = self.get_acc(argmax, l1_data_l)
        else:
            acc = 0.0
        return loss, acc

    def train_step(self, batch):
        _l, _a = self(batch, True)
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

########################################################################################################################
########################################################################################################################


class L2_CE_CLOZE(nn.Module):
    def __init__(self,
                 context_encoder,
                 l1_tied_encoder_decoder,
                 l2_tied_encoder_decoder,
                 l1_dict,
                 l2_dict,
                 l1_key,
                 l2_key,
                 l2_key_wt,
                 learn_step_regularization,
                 zero_regularization,
                 regularization_type,
                 learning_steps=3,
                 step_size=0.1,
                 max_grad_norm=5.0):
        super().__init__()
        self.context_encoder = context_encoder
        self.l1_tied_encoder_decoder = l1_tied_encoder_decoder
        self.l2_tied_encoder_decoder = l2_tied_encoder_decoder
        assert self.l2_tied_encoder_decoder.embedding_dim() == self.l1_tied_encoder_decoder.embedding_dim()
        self.l1_dict = l1_dict
        self.l1_dict_idx = {v: k for k, v in l1_dict.items()}
        self.l2_dict = l2_dict
        self.l2_dict_idx = {v: k for k, v in l2_dict.items()}
        self.l1_key = l1_key
        self.l2_key = l2_key
        self.l2_key_wt = l2_key_wt
        self.init_key()
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=self.l1_dict[SPECIAL_TOKENS.PAD])
        self.l2_exposure = {}
        self.init_param_freeze()
        self.learning_steps = learning_steps
        self.learn_step_regularization = learn_step_regularization
        self.zero_regularization = zero_regularization
        self.regularization_type = regularization_type
        assert self.learn_step_regularization >= 0.0
        assert self.zero_regularization >= 0.0
        #self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.1)
        #self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        self.max_grad_norm = max_grad_norm
        self.init_optimizer()

    def init_optimizer(self,):
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.1)
        ##self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))

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
        if self.l2_key_wt is not None:
            if self.is_cuda():
                self.l2_key_wt = self.l2_key_wt.cuda()
            else:
                pass

    def mix_inputs(self, l1_channel, l2_channel, l1_idxs, l2_idxs):
        assert l1_channel.dim() == l2_channel.dim()
        if len(l1_channel.shape) - len(l1_idxs.shape) == 0:
            v_inp_ind = l1_idxs.float() #.unsqueeze(2).expand_as(l1_channel).float()
            g_inp_ind = l2_idxs.float() #.unsqueeze(2).expand_as(l2_channel).float()
        elif len(l1_channel.shape) - len(l1_idxs.shape) == 1:
            v_inp_ind = l1_idxs.unsqueeze(2).expand_as(l1_channel).float()
            g_inp_ind = l2_idxs.unsqueeze(2).expand_as(l2_channel).float()
            pass
        else:
            raise BaseException("channel and idx mismatch by more than 1 dim!")
        encoded = (v_inp_ind * l1_channel.float() + g_inp_ind * l2_channel.float()).type_as(l1_channel)
        return encoded

    def update_l2_encoder(self, out, l2_data, l2_idxs):
        raise NotImplementedError("todo...")
        return True

    def forward(self, batch):
        lengths, l1_data, l2_data, ind, _ = batch
        l1_idxs = ind.eq(1).long()
        l2_idxs = ind.eq(2).long()
        for st in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.UNK]:  # SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.BOS]:
            if st in self.l1_dict:
                l1_idxs[l1_data.eq(self.l1_dict[st])] = 0
                l2_idxs[l1_data.eq(self.l1_dict[st])] = 0
                ind[l1_data.eq(self.l1_dict[st])] = 0
        batch_size = l1_data.size(0)
        assert batch_size == 1
        l1_encoded = self.l1_tied_encoder_decoder(l1_data, mode='input')
        l2_encoded = self.l2_tied_encoder_decoder(l2_data, mode='input')
        mixed_encoded = self.mix_inputs(l1_encoded, l2_encoded, l1_idxs, l2_idxs)
        j_data = l1_data.clone()
        j_data[l2_idxs == 1] = l2_data[l2_idxs == 1] + self.l1_tied_encoder_decoder.vocab_size()
        hidden = self.context_encoder(mixed_encoded, lengths, forward_mode='L2')
        out_l1 = self.l1_tied_encoder_decoder(hidden, mode='output')
        out_l2 = self.l2_tied_encoder_decoder(hidden, mode='output')
        l2_mask = torch.ones(out_l2.size(2)).type_as(l2_data)
        l2_mask[l2_data[0, l2_idxs[0, :] == 1]] = 0
        out_l2[:, :, l2_mask == 1] = float('-inf')
        out = torch.cat([out_l1, out_l2], dim=2)
        loss = self.loss(out.view(-1, out.size(2)), j_data.view(-1))
        return loss, l2_mask

    def regularized_step(self, l2_cpy):
        raise BaseException("not using this...")
        self.l2_tied_encoder_decoder.regularized_step(l2_cpy)
        return True

    def learn_step(self, batch):
        l2_cpy = self.l2_tied_encoder_decoder.get_word_vecs()
        ##l2_cpy = self.l2_tied_encoder_decoder.get_state_dict()
        self.init_optimizer()
        for _ in range(self.learning_steps):
            self.optimizer.zero_grad()
            loss, l2_mask = self(batch)
            #l2_regularized = ((old_l2_cpy - self.l2_tied_encoder_decoder.get_weight_without_detach()) ** 2).sum()
            l2_regularized = torch.tensor(0.0).type_as(l2_cpy)
            zero_l2_regularized = torch.tensor(0.0).type_as(l2_cpy)
            #for n, p in self.l2_tied_encoder_decoder.get_undetached_state_dict():
            for p in [self.l2_tied_encoder_decoder._compute_word_embeddings()]:
                p_changed = p[l2_mask == 0, :]
                l2_cpy_changed = l2_cpy[l2_mask == 0, :]
                if p.requires_grad and p_changed.shape[0] > 0:
                    #print('regularized_step for ', p.shape)
                    #_l2_to_previous = torch.nn.functional.mse_loss(p, l2_cpy, reduction='sum')
                    if self.regularization_type == 'mse':
                        _l2_to_previous_changed = torch.nn.functional.mse_loss(p_changed, l2_cpy_changed, reduction='sum')
                        _l2_to_zero = torch.nn.functional.mse_loss(p_changed, torch.zeros_like(l2_cpy_changed), reduction='sum')
                    elif self.regularization_type == 'huber':
                        _l2_to_previous_changed = torch.nn.functional.smooth_l1_loss(p_changed, l2_cpy_changed, reduction='sum')
                        _l2_to_zero = torch.nn.functional.smooth_l1_loss(p_changed, torch.zeros_like(l2_cpy_changed), reduction='sum')
                    elif self.regularization_type == 'l1':
                        _l2_to_previous_changed = torch.nn.functional.l1_loss(p_changed, l2_cpy_changed, reduction='sum')
                        _l2_to_zero = torch.nn.functional.l1_loss(p_changed, torch.zeros_like(l2_cpy_changed), reduction='sum')
                    else:
                        raise BaseException("unknonwn reg type")

                    l2_regularized = l2_regularized + _l2_to_previous_changed
                    zero_l2_regularized = zero_l2_regularized + _l2_to_zero
                    ##l2_regularized = l2_regularized + torch.nn.functional.mse_loss(p, torch.zeros_like(p), reduction='sum')
                else:
                    pass
            #print('%.2f' % (old_l2_regularized - l2_regularized).sum().item(), 'REGS!!!!!!!!!!!!!!!')
            final_loss = loss + (self.learn_step_regularization * l2_regularized) + \
                (self.zero_regularization * zero_l2_regularized)
            #print('x', _, final_loss.item(), loss.item(), l2_regularized.item(), zero_l2_regularized.item(), self.learn_step_regularization)
            final_loss.backward()
            #print([p.grad.sum() if p.grad is not None else 'none' for n, p in self.named_parameters()])
            grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()),
                                                       self.max_grad_norm)
            if math.isnan(grad_norm):
                print('skipping update grad_norm is nan!')
            else:
                self.optimizer.step()
        #if self.learn_step_regularization == 0.0:
        #    self.regularized_step(l2_cpy)

    def update_l2_exposure(self, l2_exposed):
        for i in l2_exposed:
            self.l2_exposure[i] = self.l2_exposure.get(i, 0.0) + 1.0

    def init_param_freeze(self,):
        for n, p in self.named_parameters():
            p.requires_grad = False
        self.l1_tied_encoder_decoder.init_param_freeze()
        self.l2_tied_encoder_decoder.init_param_freeze()
        for n, p in self.named_parameters():
            print(n, p.requires_grad)
        return True

    def init_cuda(self,):
        self.context_encoder.init_cuda()
        self.l2_tied_encoder_decoder.init_cuda()
        self.l1_tied_encoder_decoder.init_cuda()
        self = self.cuda()
        return True

    def is_cuda(self,):
        return self.context_encoder.is_cuda()

    def get_l1_word_vecs(self, on_device=True):
        return self.l1_tied_encoder_decoder.get_word_vecs(on_device)

    def get_l2_word_vecs(self, on_device=True):
        return self.l2_tied_encoder_decoder.get_word_vecs(on_device)

    def get_l2_state_dict(self, on_device=True):
        return self.l2_tied_encoder_decoder.get_state_dict()

    def set_l2_state_dict(self, state_dict):
        self.l2_tied_encoder_decoder.set_state_dict(state_dict)
        return True
