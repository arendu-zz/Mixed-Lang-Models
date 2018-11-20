import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import pdb


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self,
                 embed_dim,
                 encoder_ffn_embed_dim,
                 attn_heads,
                 attn_dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim  # args.encoder_embed_dim
        self.attn_dropout = attn_dropout
        #self.self_attn = MultiheadAttention(
        #    self.embed_dim, args.encoder_attention_heads,
        #    dropout=args.attention_dropout,
        #)
        #self.self_attn = MultiheadAttention(
        #    self.embed_dim, attn_heads,
        #    dropout=attn_dropout,
        #)
        self.self_attn = StupidAttention(self.embed_dim, self.attn_dropout)
        #self.relu_dropout = relu_dropout
        self.layer_norm = LayerNorm(self.embed_dim)
        #self.fc1 = Linear(self.embed_dim, encoder_ffn_embed_dim)
        #self.fc2 = Linear(encoder_ffn_embed_dim, self.embed_dim)
        #self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        # residual = x
        x, attn_probs = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.attn_dropout, training=self.training)
        x = self.layer_norm(x)  # LAYER NORM SEEMS IMPORTANT
        # x = residual + x

        #residual = x
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=self.relu_dropout, training=self.training)
        #x = self.fc2(x)
        #x = F.dropout(x, p=self.dropout, training=self.training)
        #x = residual + x
        #x = self.maybe_layer_norm(1, x, after=True)
        return x, attn_probs

    def maybe_layer_norm(self, i, x):
        return self.layer_norms[i](x)


def Linear(in_features, out_features, bias=True, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


class StupidAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_weight_dropout = nn.Dropout(dropout)
        self.rand_attn_weight_dropout = nn.Dropout(dropout)
        #self.q_linear = Linear(self.embed_dim, self.embed_dim)
        self.k_linear = Linear(self.embed_dim, self.embed_dim)
        #self.v_linear = Linear(self.embed_dim, self.embed_dim)
        #self.out_proj = Linear(self.embed_dim, self.embed_dim)

    def forward(self, query, key, value, key_padding_mask):
        bsz, src_len, embed_dim = query.shape
        q = query  # self.q_linear(query)
        k = self.k_linear(key)
        v = value
        #q = self.q_linear(query).transpose(0, 1)
        #q = query.transpose(0, 1) #self.q_linear(query).transpose(0, 1)
        #k = self.k_linear(key).transpose(0, 1)
        #k = key.transpose(0, 1)
        #v = self.v_linear(value).transpose(0, 1)
        #v = value.transpose(0, 1) #self.v_linear(value).transpose(0, 1)
        #q = self.q_linear(query).transpose(0, 1)
        #k = self.k_linear(key).transpose(0, 1)
        #v = self.v_linear(value).transpose(0, 1)
        # bsz, src_len, embed_dim = q.shape == k.shape == v.shape
        attn_weights = torch.bmm(q, k.transpose(2, 1)) * (1.0 / math.sqrt(embed_dim)) #SCALING SEEMS VERY IMPORTANT
        # bsz, src_len, src_len = attn_weights.shape
        assert key_padding_mask is not None
        #if key_padding_mask is not None:
            #key_padding_mask = key_padding_mask.unsqueeze(1)  # .unsqueeze(2)
        #attn_weights = attn_weights.float().masked_fill(key_padding_mask, float('-inf')).type_as(attn_weights)  # FP16 support: cast to float and back
        attn_weights = attn_weights.float().masked_fill(key_padding_mask, float('-inf')).type_as(attn_weights)  # FP16 support: cast to float and back
        #pdb.set_trace()
        #rand_mask = self.rand_attn_weight_dropout(torch.ones_like(attn_weights).type_as(attn_weights))
        #attn_weights = attn_weights.float().masked_fill(rand_mask.eq(0), -1e8).type_as(attn_weights)
        attn_weights = attn_weights.view(bsz, src_len, src_len)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        #pdb.set_trace()
        #print(attn_weights[0, :, :].argmax(1), attn_weights[0, :, :].sum(1).sum().item(), src_len)
        attn = torch.bmm(attn_weights, v)
        #attn = attn.transpose(0, 1).contiguous().view(src_len, bsz, embed_dim)
        #attn = self.out_proj(attn)
        return attn, attn_weights

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self._mask = None

        self.in_proj_weight = Parameter(torch.Tensor(3*embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3*embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        #self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        #nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            #nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, mask_future_timesteps=False,
                key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()
        
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
            #seq_len, bs, emb_size = q.shape == k.shape == v.shape
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                # this will allow us to concat it with previous value and get
                # just get the previous value
                k = v = q.new(0)
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if saved_state is not None:
            if 'prev_key' in saved_state:
                k = torch.cat((saved_state['prev_key'], k), dim=0)
            if 'prev_value' in saved_state:
                v = torch.cat((saved_state['prev_value'], v), dim=0)
            saved_state['prev_key'] = k
            saved_state['prev_value'] = v
            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
            assert key_padding_mask.size(2) == src_len  # added by arendu

        q = q.contiguous().view(tgt_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        #bs, seq_len, emb_size = q.shape == k.shape == v.shape
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # only apply masking at training time (when incremental state is None)
        if mask_future_timesteps and incremental_state is None:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += self.buffered_mask(attn_weights).unsqueeze(0)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            key_padding_mask = key_padding_mask.unsqueeze(1)  # .unsqueeze(2)
            attn_weights = attn_weights.float().masked_fill(key_padding_mask, float('-inf'),).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        #print(attn_weights[0, :, :].argmax(1), attn_weights[0, :, :].sum(1).sum().item(), src_len)
        #print(attn_weights[-1, :, :].argmax(1), attn_weights[-1, :, :].sum(1))
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        #attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None
        #seq_len, bs, emb_size = attn.shape
        #bs, seq_len, seq_len = attn_weights.shape
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2*self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2*self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(fill_with_neg_inf(self._mask.resize_(dim, dim)), 1)
        return self._mask[:dim, :dim]
