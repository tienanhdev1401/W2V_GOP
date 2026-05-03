from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn


def _no_grad_trunc_normal_(tensor: torch.Tensor, mean: float, std: float, a: float, b: float) -> torch.Tensor:
    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, steps, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, steps, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Wav2VecGOPT(nn.Module):
    """
    GOPT variant with an input adapter for wav2vec/prosody token features.
    Input/output signatures are kept compatible with the original GOPT pipeline.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        input_dim: int = 770,
        adapter_dim: int = 256,
        adapter_dropout: float = 0.1,
        max_seq_len: int = 50,
        use_phn_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_phn_embedding = use_phn_embedding

        self.input_adapter = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(adapter_dropout),
            nn.Linear(adapter_dim, embed_dim),
        )

        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])

        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len + 5, self.embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        if self.use_phn_embedding:
            self.phn_proj = nn.Linear(40, embed_dim)

        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token5 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        trunc_normal_(self.cls_token1, std=0.02)
        trunc_normal_(self.cls_token2, std=0.02)
        trunc_normal_(self.cls_token3, std=0.02)
        trunc_normal_(self.cls_token4, std=0.02)
        trunc_normal_(self.cls_token5, std=0.02)

    def forward(self, x: torch.Tensor, phn: torch.Tensor):
        bsz, seq_len, _ = x.shape

        if seq_len + 5 > self.pos_embed.shape[1]:
            raise ValueError(f"Input sequence length {seq_len:d} exceeds max_seq_len {self.max_seq_len:d}")

        x = self.input_adapter(x)

        if self.use_phn_embedding:
            phn_one_hot = torch.nn.functional.one_hot(phn.long() + 1, num_classes=40).float()
            phn_embed = self.phn_proj(phn_one_hot)
            x = x + phn_embed

        cls_token1 = self.cls_token1.expand(bsz, -1, -1)
        cls_token2 = self.cls_token2.expand(bsz, -1, -1)
        cls_token3 = self.cls_token3.expand(bsz, -1, -1)
        cls_token4 = self.cls_token4.expand(bsz, -1, -1)
        cls_token5 = self.cls_token5.expand(bsz, -1, -1)

        x = torch.cat((cls_token1, cls_token2, cls_token3, cls_token4, cls_token5, x), dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]

        for blk in self.blocks:
            x = blk(x)

        u1 = self.mlp_head_utt1(x[:, 0])
        u2 = self.mlp_head_utt2(x[:, 1])
        u3 = self.mlp_head_utt3(x[:, 2])
        u4 = self.mlp_head_utt4(x[:, 3])
        u5 = self.mlp_head_utt5(x[:, 4])

        p = self.mlp_head_phn(x[:, 5:])
        w1 = self.mlp_head_word1(x[:, 5:])
        w2 = self.mlp_head_word2(x[:, 5:])
        w3 = self.mlp_head_word3(x[:, 5:])
        return u1, u2, u3, u4, u5, p, w1, w2, w3
