import math
import torch
import torch.nn as nn
from src.utils import subsequent_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,T,D)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, key_keep_mask=None):
        B, Tq, _ = q.size()
        _, Tk, _ = k.size()

        Q = self.wq(q).view(B, Tq, self.n_heads, self.d_head).transpose(1,2)  
        K = self.wk(k).view(B, Tk, self.n_heads, self.d_head).transpose(1,2)
        V = self.wv(v).view(B, Tk, self.n_heads, self.d_head).transpose(1,2)

        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.d_head)  

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))

        if key_keep_mask is not None:
            scores = scores.masked_fill(~key_keep_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = attn @ V  
        out = out.transpose(1,2).contiguous().view(B, Tq, self.d_model)
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x): return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_keep_mask):
        a = self.attn(x, x, x, key_keep_mask=src_keep_mask)
        x = self.ln1(x + self.drop(a))
        f = self.ff(x)
        x = self.ln2(x + self.drop(f))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_attn_mask, src_keep_mask, tgt_keep_mask):
        a = self.self_attn(x, x, x, attn_mask=tgt_attn_mask, key_keep_mask=tgt_keep_mask)
        x = self.ln1(x + self.drop(a))
        c = self.cross_attn(x, memory, memory, key_keep_mask=src_keep_mask)
        x = self.ln2(x + self.drop(c))
        f = self.ff(x)
        x = self.ln3(x + self.drop(f))
        return x

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.src_emb = nn.Embedding(src_vocab_size, cfg.d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.dropout, max_len=1000)
        self.enc = nn.ModuleList([EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                  for _ in range(cfg.n_layers)])
        self.dec = nn.ModuleList([DecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                  for _ in range(cfg.n_layers)])
        self.out = nn.Linear(cfg.d_model, tgt_vocab_size)

    def encode(self, src, src_keep_mask):
        x = self.src_emb(src) * math.sqrt(self.cfg.d_model)
        x = self.pos(x)
        for layer in self.enc:
            x = layer(x, src_keep_mask)
        return x

    def forward(self, src, tgt_inp, src_keep_mask, tgt_keep_mask):
        memory = self.encode(src, src_keep_mask)
        T = tgt_inp.size(1)
        tgt_attn_mask = subsequent_mask(T).to(tgt_inp.device)  
        x = self.tgt_emb(tgt_inp) * math.sqrt(self.cfg.d_model)
        x = self.pos(x)
        for layer in self.dec:
            x = layer(x, memory, tgt_attn_mask, src_keep_mask, tgt_keep_mask)
        return self.out(x)
