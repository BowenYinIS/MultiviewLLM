import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ProjectorLayer(nn.Module):
    """
    x: (B, T, input_dim)  ->  out: (B, Q, output_dim)
    Learnable queries: (Q, hidden_dim)
    Cross-attn: Q x (K,V(x))
    """
    def __init__(
        self,
        query_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        prenorm: bool = True,
        ffw_ratio: float = 2.0,   # 小型FFN，可关
    ):
        super().__init__()
        self.query_num   = query_num
        self.hidden_dim  = hidden_dim
        self.output_dim  = output_dim or hidden_dim
        self.num_heads   = num_heads

        # 可学习 query（Q×H）
        self.query = nn.Parameter(torch.randn(query_num, hidden_dim) * (hidden_dim ** -0.5))

        # 线性映射得到 K,V（T×H）
        self.k_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        # 多头注意力（batch_first=True: (B, L, H)）
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 轻量FFN + 输出投影（可选）
        self.prenorm = prenorm
        self.norm_q  = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        ffw_hidden   = int(ffw_ratio * hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffw_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffw_hidden, hidden_dim),
        ) if ffw_ratio and ffw_ratio > 0 else nn.Identity()

        self.out_proj = nn.Identity() if self.output_dim == hidden_dim else nn.Linear(hidden_dim, self.output_dim)
        self.drop = nn.Dropout(dropout)

        # 初始化线性层（线性层自带kaiming，但这里更稳）
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if isinstance(self.out_proj, nn.Linear):
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,                 # (B, T, input_dim)
        attn_mask: Optional[torch.Tensor] = None,  # (B, T) 1=keep, 0=pad
        return_attn: bool = False,
    ):
        B, T, _ = x.shape

        K = self.k_proj(x)  # (B, T, H)
        V = self.v_proj(x)  # (B, T, H)

        # 构造 batch 维度上的 queries: (B, Q, H)
        Q = self.query.unsqueeze(0).expand(B, -1, -1)

        # 预归一化（更稳）
        if self.prenorm:
            Q = self.norm_q(Q)
            K = self.norm_kv(K)

        key_padding_mask = None
        if attn_mask is not None:
            # MultiheadAttention 里：True=mask（要忽略），False=keep
            key_padding_mask = (attn_mask == 0)  # (B, T) bool

        # cross-attn: query=Q, key=K, value=V
        out, attn = self.attn(
            query=Q, key=K, value=V,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,  # 返回每头权重
        )  # out: (B, Q, H)

        # 残差 + FFN
        out = out + self.drop(self.ffn(out))  # (B, Q, H)

        # 输出投影
        out = self.out_proj(out)  # (B, Q, output_dim)

        if return_attn:
            return out, attn  # attn: (B, num_heads, Q, T)
        return out


class Projector(nn.Module):
    def __init__(self,
        llm_embed: Optional[nn.Embedding],
        query_num: int = 8,
        graph_input_dim: int = 128,
        ts_input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        prenorm: bool = True,
        ffw_ratio: float = 2.0,
    ):
        super().__init__()
        self.graph_projector = ProjectorLayer(
            query_num=query_num,
            input_dim=graph_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            prenorm=prenorm,
            ffw_ratio=ffw_ratio,
        )
        self.ts_projector = ProjectorLayer(
            query_num=query_num,
            input_dim=ts_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            prenorm=prenorm,
            ffw_ratio=ffw_ratio,
        )

        self.llm_embed = llm_embed
        if self.llm_embed:
            for param in self.llm_embed.parameters():
                param.requires_grad = False

    @staticmethod
    def _scatter_slots(inputs_embeds, where_mask, slots):
        # inputs_embeds: (B, L, D)
        # where_mask:    (B, L) bool
        # slots:         (B, Q, D)
        B, L, D = inputs_embeds.shape
        Q = slots.size(1)

        # 避免潜在 in-place autograd 冲突
        out = inputs_embeds.clone()

        for b in range(B):
            pos = torch.nonzero(where_mask[b], as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                continue
            # n = min(pos.numel(), Q)
            assert pos.numel() == Q, "Number of slots must match number of positions to replace."
            out[b, pos, :] = slots[b, :, :]
        return out

    def forward(
        self,
        input_ids: Optional[torch.Tensor],  # (B, L)
        is_graph: Optional[torch.Tensor] = None,  # (B, L) bool
        is_ts: Optional[torch.Tensor] = None,     # (B, L) bool
        graph_x: Optional[torch.Tensor] = None,    # (B, T_g, d_g)
        graph_x_pad: Optional[torch.Tensor] = None,  # (B, T_g) 1=keep, 0=pad
        ts_x: Optional[torch.Tensor] = None,       # (B, T_t, d_t)
        ts_x_pad: Optional[torch.Tensor] = None,   # (B, T_t) 1=keep, 0=pad
    ):
        # 获得LLM的token embedding
        input_embeds = self.llm_embed(input_ids)  # (B, L, d_l)

        # 获得嵌入embedding
        if graph_x is not None:
            graph_embeds = self.graph_projector(
                graph_x,
                attn_mask=graph_x_pad,
                return_attn=False,
            )  # (B, Q, output_dim)

            # change graph_embeds dtype to match input_embeds dtype
            graph_embeds = graph_embeds.to(input_embeds.dtype)

            input_embeds = self._scatter_slots(
                input_embeds,
                is_graph,
                graph_embeds,
            )

        if ts_x is not None:
            ts_embeds = self.ts_projector(
                ts_x,
                attn_mask=ts_x_pad,
                return_attn=False,
            )

            ts_embeds = ts_embeds.to(input_embeds.dtype)

            input_embeds = self._scatter_slots(
                input_embeds,
                is_ts,
                ts_embeds,
            )

        return input_embeds




