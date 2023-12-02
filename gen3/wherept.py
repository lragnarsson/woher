from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import PADDING_TOKEN, END_TOKEN


@dataclass
class WherePTConfig:
    vocab_len: int = 61,
    n_embed: int = 128
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 8
    dropout: float = 0.1


class MaskedTensor:
    def __init__(self, tensor, mask):
        self.tensor = tensor
        self.mask = mask


class CausalSelfAttentionHead(nn.Module):

    def __init__(self, config: WherePTConfig, head_size: int):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(config.n_embed, head_size, bias=False)
        self.key = nn.Linear(config.n_embed, head_size, bias=False)
        self.value = nn.Linear(config.n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x, padding_mask=None):
        B, T, C = x.shape # (batch_size, seq_len, config.n_embed)

        query = self.query(x) # (B, T, C)
        key = self.key(x) # (B, T, C)

        # Compute attention scores:
        wei = query @ key.transpose(-2, -1) * C**(-0.5) # (B, T, C) @ (B, C, T) = (B, T, T)
        # Mask out future tokens:
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        # Mask out padding tokens:
        if padding_mask is not None:
           wei = wei.masked_fill(padding_mask.unsqueeze(1).expand(-1, T, -1), float("-1e9"))
        
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, C)
        return wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
    
class MultiHeadAttention(nn.Module):

    def __init__(self, config: WherePTConfig, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([CausalSelfAttentionHead(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, padding_mask=None):
        out = torch.cat([h(x, padding_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):

    def __init__(self, config: WherePTConfig):
        super().__init__()
        head_size = config.n_embed // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedForward(config.n_embed)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x: MaskedTensor):
        x.tensor = self.sa(self.ln1(x.tensor), x.mask)
        x.tensor = self.ffwd(self.ln2(x.tensor))
        return x

class WherePT(nn.Module):

    def __init__(self, config: WherePTConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_len, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)

        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_len)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        padding_mask = (idx == PADDING_TOKEN) # (batch_size, seq_len)

        tok_embeds = self.token_embedding(idx) # (batch_size, seq_len, n_embed)
        pos_embeds = self.position_embedding(torch.arange(T)) # (seq_len, n_embed)
        x = MaskedTensor(tok_embeds + pos_embeds, padding_mask) # (batch_size, seq_len, n_embed)
        x = self.blocks(x)
        x.tensor = self.ln_final(x.tensor)
        logits = self.lm_head(x.tensor) # (batch_size, seq_len, vocab_len)  

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_len), targets.view(-1))

        return logits, loss
    
    def generate(self, idx, max_new_tokens=10):
        # idx: (batch_size, seq_len)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:] # (batch_size, block_size)
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if next_token == END_TOKEN:
                break
        return idx