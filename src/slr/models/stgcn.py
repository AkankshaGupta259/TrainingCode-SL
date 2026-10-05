"""Native ST-GCN (Yan et al., AAAI 2018) for isolated SLR.

This is the fully-implemented reference skeleton model. Input is a batch of
skeleton sequences shaped ``(N, C, T, V, M)`` and the output is class logits.
Other GCNs plug into the same registry (see ``slr.models``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.graph import Graph


class GraphConv(nn.Module):
    """Spatial graph convolution (K partitions)."""

    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(in_channels, out_channels * k, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                               # N, out*k, T, V
        n, kc, t, v = x.size()
        x = x.view(n, self.k, kc // self.k, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, A)      # aggregate over partitions
        return x.contiguous()


class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1, residual=True, dropout=0.0):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c, k)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout, inplace=True),
        )
        self.bn_gcn = nn.BatchNorm2d(out_c)
        if not residual:
            self.residual = lambda x: 0
        elif in_c == out_c and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, (stride, 1)),
                nn.BatchNorm2d(out_c),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.bn_gcn(self.gcn(x, A))
        x = self.tcn(x) + res
        return self.relu(x)


class STGCN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        graph_strategy: str = "spatial",
        edge_importance: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        graph = Graph(strategy=graph_strategy)
        A = torch.tensor(graph.A, dtype=torch.float32)
        self.register_buffer("A", A)
        k = A.size(0)
        V = A.size(1)

        self.data_bn = nn.BatchNorm1d(in_channels * V)
        cfgs = [
            (in_channels, 64, 1, False),
            (64, 64, 1, True), (64, 64, 1, True), (64, 64, 1, True),
            (64, 128, 2, True), (128, 128, 1, True), (128, 128, 1, True),
            (128, 256, 2, True), (256, 256, 1, True), (256, 256, 1, True),
        ]
        self.blocks = nn.ModuleList(
            [STGCNBlock(i, o, k, stride=s, residual=r, dropout=dropout) for i, o, s, r in cfgs]
        )
        if edge_importance:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones_like(A)) for _ in self.blocks]
            )
        else:
            self.edge_importance = [1] * len(self.blocks)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for block, imp in zip(self.blocks, self.edge_importance):
            x = block(x, self.A * imp)

        x = F.adaptive_avg_pool2d(x, 1).view(N, M, -1).mean(dim=1)  # (N, 256)
        return self.fc(x)
