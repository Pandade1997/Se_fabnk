import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math, copy
from util.stft_istft import STFT
import config


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# ******************************************
class SublayerConnection(nn.Module):  # Add & Normalize & attention
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # sublayer: attention or feed-forward


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # 在最后一层Encoder后加一层LayerNorm 因为EncoderLayer的输出是未经LayerNorm的 所以需要在这里处理


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # *******************************************************************
        # sublayer[0]完成了 LayerNorm(x)、self attention、残差(x+z) 三个任务
        # 每一层的输入（或者说输出）都是未经LayerNorm的 所以要先对输入做LayerNorm
        # lambda... 相当于def func(x):self_attn(x, x, x, mask)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        # sublayer[1]完成了 LayerNorm(x+z)、feed_forward、残差((x+z)+z`) 三个任务
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # 为什么是tgt_mask
        x = self.sublayer[1](x, lambda x: self.self_attn(x, m, m, src_mask))  # 为什么是src_mask  而且m好像没有LayerNorm
        return self.sublayer[2](x, self.feed_forward)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    # np.triu 返回函数的上三角矩阵 k=0包括对角线;k=1不包括对角线;k=-1包括对角线向下平移一个单位及以上的数据
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype(np.uint8)
    return torch.from_numpy(subsequent_mask) == 0  # 返回一个bool矩阵


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query.transpose(1, 2), key.transpose(1, 2).transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # ?

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:  # dropout
        p_attn = dropout(p_attn)

    attn = torch.matmul(p_attn, value.transpose(1, 2))
    return attn, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Conv1d(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        #  ************************************************************************
        # q,k,v shape: nbatches * x(-1表示该位置由其他位置推断) * head数 * d_k(d_model / h)
        # 这里的的qkv是经过线性层变换后的qkv 即乘了变化矩阵W后的qkv
        # 为每个head输入的长度为d_k而不是d_model 所以每个head得到的信息应该是不全的
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k)
                             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):  # 编码长度最大支持5000
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 从0开始给偶数位置编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 从1开始给奇数位置编码
        pe = pe.unsqueeze(0)
        pe_ = pe
        for i in range(32 - 1):  # 最多可以处理多少个batch 需要调大batch时修改 *************
            pe = torch.cat((pe, pe_), 0)

        self.register_buffer('pe', pe)  # pe为1*5000*20 5000由max_len决定 20由d_model决定

    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0), :x.size(1), :], requires_grad=False)
        return self.dropout(x)


# 在这里对输入输出进行操作
class PosEncoder(nn.Module):
    def __init__(self, encoder, pos):
        super(PosEncoder, self).__init__()
        self.encoder = encoder
        self.pos = pos
        self.stft = STFT(config.WIN_LEN, config.WIN_OFFSET).cuda()

    def forward(self, src, src_mask=None):
        # 频域
        magnitude, phase = self.stft.transform(src)
        magnitude = magnitude.permute(0, 2, 1)
        est = self.encode(magnitude, src_mask)

        # 时域
        # feat = self.ola.transform(src) # 分帧操作 可以用其他方式分帧
        # est = self.encode(feat, src_mask)
        return est  # 这里就是整个模型的输出

    def encode(self, src, src_mask):
        x = self.pos(src)
        return self.encoder(x, src_mask)


# 入口函数
# ff:feed forward   d_ff:ff的一层的输出，第二层的输入    d_model:输入      h:多头个数
# def make_long_model(n=args.ENCODER_NUM, d_model=config.WIN_LEN // 2, d_ff=2048, h=config.HEAD, dropout=0.1):
def make_long_model(n=12, d_model=None, d_ff=120, h=6, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = PosEncoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n), c(position))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
