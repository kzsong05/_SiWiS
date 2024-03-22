import math
import torch
from torch import nn
from einops import rearrange


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        act_fn: bool = True,
        dropout: float = 0.1,
    ):
        super(LinearLayer, self).__init__()
        self.proj = nn.Linear(in_dims, out_dims)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dims)
        self.act_fn = nn.ReLU() if act_fn else nn.Identity()

    def forward(self, hidden_input):
        hidden_state = self.proj(hidden_input)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.act_fn(hidden_state)
        return hidden_state

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(MultiHeadSelfAttention, self).__init__()

        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-8)

    def forward(self, hidden_input):
        hidden_query = self.query(hidden_input)
        hidden_query = rearrange(hidden_query, "n l (h d) -> n h l d", h=self.num_heads)
        hidden_key = self.key(hidden_input)
        hidden_key = rearrange(hidden_key, "n l (h d) -> n h l d", h=self.num_heads)
        hidden_value = self.value(hidden_input)
        hidden_value = rearrange(hidden_value, "n l (h d) -> n h l d", h=self.num_heads)

        attention_weights = torch.matmul(hidden_query, hidden_key.transpose(-1, -2))
        attention_weights = nn.functional.softmax(attention_weights / math.sqrt(self.head_dim), dim=-1)
        attention_weights = self.dropout(attention_weights)

        hidden_attn = torch.matmul(attention_weights, hidden_value)
        hidden_attn = rearrange(hidden_attn, "n h l d -> n l (h d)")

        hidden_state = self.proj(hidden_attn)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.norm(hidden_state + hidden_input)
        return hidden_state


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(MultiHeadCrossAttention, self).__init__()

        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-8)

    def forward(self, hidden_input, cross_hidden_input):
        hidden_query = self.query(hidden_input)
        hidden_query = rearrange(hidden_query, "n v l (h d) -> n v h l d", h=self.num_heads)
        hidden_key = self.key(cross_hidden_input)
        hidden_key = rearrange(hidden_key, "n v l (h d) -> n v h l d", h=self.num_heads)
        hidden_value = self.value(cross_hidden_input)
        hidden_value = rearrange(hidden_value, "n v l (h d) -> n v h l d", h=self.num_heads)

        attention_weights = torch.matmul(hidden_query, hidden_key.transpose(-1, -2))
        attention_weights = nn.functional.softmax(attention_weights / math.sqrt(self.head_dim), dim=-1)
        attention_weights = self.dropout(attention_weights)

        hidden_attn = torch.matmul(attention_weights, hidden_value)
        hidden_attn = rearrange(hidden_attn, "n v h l d -> n v l (h d)")

        hidden_state = self.proj(hidden_attn)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.norm(hidden_state + hidden_input)
        return hidden_state


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim * 2, eps=1e-8)
        self.act_fn1 = nn.GELU()

        self.linear2 = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-8)
        self.act_fn2 = nn.GELU()

    def forward(self, hidden_input):
        hidden_state = self.linear1(hidden_input)
        hidden_state = self.dropout1(hidden_state)
        hidden_state = self.norm1(hidden_state)
        hidden_state = self.act_fn1(hidden_state)

        hidden_state = self.linear2(hidden_state)
        hidden_state = self.dropout2(hidden_state)
        hidden_state = self.norm2(hidden_state + hidden_input)
        hidden_state = self.act_fn2(hidden_state)
        return hidden_state


class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        max_seq_length: int,
        hidden_embed_dim: int,
        dropout: float = 0.1
    ):
        super(SelfAttnBlock, self).__init__()
        assert in_dims == out_dims
        self.max_seq_length = max_seq_length

        self.proj_in = LinearLayer(in_dims, hidden_embed_dim, act_fn=False)
        self.pos_emb = nn.Embedding(self.max_seq_length, hidden_embed_dim)
        self.emb_norm = nn.LayerNorm(hidden_embed_dim)

        self.self_attn = MultiHeadSelfAttention(embed_dim=hidden_embed_dim)
        self.feed_forward = FeedForward(embed_dim=hidden_embed_dim)

        self.proj_out = nn.Linear(hidden_embed_dim, out_dims)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dims)
        self.act_fn = nn.ReLU()

    def forward(self, hidden_input):
        hidden_state = self.proj_in(hidden_input)

        p_ids = torch.arange(self.max_seq_length, dtype=torch.int, device=hidden_input.device)
        pe = self.pos_emb(p_ids).unsqueeze(0)
        hidden_state = self.emb_norm(pe + hidden_state)

        hidden_state = self.self_attn(hidden_state)
        hidden_state = self.feed_forward(hidden_state)

        hidden_state = self.proj_out(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.norm(hidden_state + hidden_input)
        hidden_state = self.act_fn(hidden_state)
        return hidden_state


class PixelAttnBlock(nn.Module):
    def __init__(
        self,
        in_dims: int,
        hidden_embed_dim: int,
        max_pixel_length: int,
        max_frame_length: int,
    ):
        super(PixelAttnBlock, self).__init__()

        self.max_frame_length = max_frame_length
        self.max_pixel_length = max_pixel_length

        self.proj_in = LinearLayer(in_dims, hidden_embed_dim, act_fn=False)
        self.pixel_embedding = nn.Embedding(self.max_pixel_length, hidden_embed_dim)
        self.pixel_norm = nn.LayerNorm(hidden_embed_dim)

        self.self_attn = MultiHeadSelfAttention(embed_dim=hidden_embed_dim)
        self.feed_forward = FeedForward(embed_dim=hidden_embed_dim)

    def forward(self, hidden_input):
        hidden_state = self.proj_in(hidden_input).unsqueeze(2)

        pixel_ids = torch.arange(self.max_pixel_length, dtype=torch.int, device=hidden_input.device)
        pixel_embed = self.pixel_embedding(pixel_ids).unsqueeze(0).unsqueeze(0)

        hidden_state = self.pixel_norm(hidden_state + pixel_embed)
        hidden_state = rearrange(hidden_state, "n v l d -> (n v) l d")
        hidden_state = self.self_attn(hidden_state)
        hidden_state = rearrange(hidden_state, "(n v) l d -> n v l d", v=self.max_frame_length)
        hidden_state = self.feed_forward(hidden_state)

        return hidden_state
