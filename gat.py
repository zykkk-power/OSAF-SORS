"""Graph-attention components used by the OSAF-SORS model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """A sparse multi-head graph-attention layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        heads: int = 1,
        dropout: float = 0.2,
        alpha: float = 0.2,
        concat: bool = True,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout
        self.concat = concat

        self.weight = nn.Parameter(torch.empty(heads, in_features, out_features))
        self.attn_src = nn.Parameter(torch.empty(heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_features))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(-1), gain=1.414)
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(-1), gain=1.414)

    @staticmethod
    def _with_self_loops(edge_index: Tensor, node_count: int) -> Tensor:
        device = edge_index.device
        loops = torch.arange(node_count, device=device, dtype=torch.long)
        return torch.cat((edge_index.long(), torch.stack((loops, loops))), dim=1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if x.dim() != 2:
            raise ValueError("GraphAttentionLayer expects [nodes, features]")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, edge_count]")

        node_count = x.size(0)
        edge_index = self._with_self_loops(edge_index.to(x.device), node_count)
        source, destination = edge_index
        # [nodes, heads, output_features]
        projected = torch.einsum("nf,hfo->nho", x, self.weight)
        logits = self.leaky_relu(
            (projected[source] * self.attn_src).sum(-1)
            + (projected[destination] * self.attn_dst).sum(-1)
        )

        # Normalise attention separately over every destination neighbourhood.
        # The node loop deliberately keeps this implementation dependency-free
        # (no torch_scatter requirement) while retaining autograd support.
        outputs: list[Tensor] = []
        for node in range(node_count):
            mask = destination == node
            node_logits = logits[mask]  # [incoming_edges, heads]
            attention = F.softmax(node_logits, dim=0)
            attention = F.dropout(attention, p=self.dropout, training=self.training)
            outputs.append((attention.unsqueeze(-1) * projected[source[mask]]).sum(dim=0))
        output = torch.stack(outputs, dim=0)
        # Apply sigma to every head before the head outputs are concatenated or
        # averaged. ELU is the standard GAT activation.
        output = F.elu(output)
        if self.concat:
            return output.reshape(node_count, self.heads * self.out_features)
        return output.mean(dim=1)


class MMGAT(nn.Module):
    """Three-layer hierarchical multi-head GAT."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 28,
        output_dim: int = 28,
        heads: int = 8,
        dropout: float = 0.2,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.layer1 = GraphAttentionLayer(
            input_dim, hidden_dim, heads=heads, dropout=dropout, alpha=alpha, concat=True
        )
        self.layer2 = GraphAttentionLayer(
            hidden_dim * heads,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            alpha=alpha,
            concat=True,
        )
        self.layer3 = GraphAttentionLayer(
            hidden_dim * heads,
            output_dim,
            heads=1,
            dropout=dropout,
            alpha=alpha,
            concat=False,
        )
        self.residual1 = nn.Linear(input_dim, hidden_dim * heads, bias=False)
        self.residual2 = nn.Linear(hidden_dim * heads, hidden_dim * heads, bias=False)
        self.residual3 = nn.Linear(hidden_dim * heads, output_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim * heads)
        self.norm3 = nn.LayerNorm(output_dim)

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        x = F.dropout(features, p=self.dropout, training=self.training)
        x = self.norm1(self.layer1(x, edge_index) + self.residual1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm2(self.layer2(x, edge_index) + self.residual2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm3(self.layer3(x, edge_index) + self.residual3(x))
        return x
