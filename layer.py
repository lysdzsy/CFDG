import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

class InstrumentalVariableModule(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, z_dim)
        self.logvar_layer = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        h = self.encoder(xy)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class multi_shallow_embedding(nn.Module):

    def __init__(self, num_nodes, k_neighs, num_graphs):
        super().__init__()

        self.num_nodes = num_nodes
        self.k = k_neighs
        self.num_graphs = num_graphs

        self.emb_s = Parameter(Tensor(num_graphs, num_nodes, 1))
        self.emb_t = Parameter(Tensor(num_graphs, 1, num_nodes))

    def reset_parameters(self):
        init.xavier_uniform_(self.emb_s)
        init.xavier_uniform_(self.emb_t)

    def forward(self, device):
        adj = torch.matmul(self.emb_s, self.emb_t).to(device)
        adj[:, torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = float('-inf')
        adj_mask = torch.zeros_like(adj)

        for g in range(self.num_graphs):
            for i in range(self.num_nodes):
                topk = torch.topk(adj[g, i], k=self.k)[1]
                topk = topk[topk != i]
                adj_mask[g, i, topk] = 1.0

        adj = adj_mask
        return adj


class LGALearner(nn.Module):
    def __init__(self, emb_dim, attention_dim=64):
        super(LGALearner, self).__init__()
        self.input_dim = emb_dim
        self.attention_dim = attention_dim

        self.attn_weight = nn.Parameter(torch.Tensor(emb_dim, attention_dim))
        self.attn_query = nn.Parameter(torch.Tensor(emb_dim, attention_dim))

        self.attn_proj_layer = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.attn_weight)
        nn.init.xavier_uniform_(self.attn_query)
        for layer in self.attn_proj_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, node_emb, edge_index):
        src, dst = edge_index[0], edge_index[1]
        emb_src = node_emb[src]  # [num_edges, emb_dim]
        emb_dst = node_emb[dst]  # [num_edges, emb_dim]

        src_proj = torch.matmul(emb_src, self.attn_weight)
        dst_proj = torch.matmul(emb_dst, self.attn_query)
        proj_sum = src_proj + dst_proj
        attn_score = self.attn_proj_layer(proj_sum).squeeze(-1)  # shape: [num_edges]
        return attn_score


class Group_Linear(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super().__init__()

        self.out_channels = out_channels
        self.groups = groups

        self.group_mlp = nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size=(1, 1), groups=groups,
                                   bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.group_mlp.reset_parameters()

    def forward(self, x: Tensor, is_reshape: False):
        """
        Args:
            x (Tensor): [B, C, N, F] (if not is_reshape), [B, C, G, N, F//G] (if is_reshape)
        """
        B = x.size(0)
        C = x.size(1)
        N = x.size(-2)
        G = self.groups

        if not is_reshape:
            # x: [B, C_in, G, N, F//G]
            x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        # x: [B, G*C_in, N, F//G]
        x = x.transpose(1, 2).reshape(B, G * C, N, -1)

        out = self.group_mlp(x)
        out = out.reshape(B, G, self.out_channels, N, -1).transpose(1, 2)

        # out: [B, C_out, G, N, F//G]
        return out


class DenseGINConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, eps=0, train_eps=True):
        super().__init__()

        # TODO: Multi-layer model
        self.mlp = Group_Linear(in_channels, out_channels, groups, bias=False)

        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(Tensor([eps]))
        else:
            self.register_buffer('eps', Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.init_eps)

    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[..., idx, idx] += 1

        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        return adj

    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        B, C, N, _ = x.size()
        G = adj.size(0)

        # adj-norm
        adj = self.norm(adj, add_loop=False)

        # x: [B, C, G, N, F//G]
        x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        # print(adj.shape)########torch.Size([4, 4, 4])
        # print(x.shape)##########torch.Size([16, 256, 4, 4, 256])

        out = torch.matmul(adj, x)

        # # DYNAMIC
        x_pre = x[:, :, :-1, ...]
        out[:, :, 1:, ...] = out[:, :, 1:, ...] + x_pre

        if add_loop:
            out = (1 + self.eps) * x + out

        # out: [B, C, G, N, F//G]
        out = self.mlp(out, True)

        # out: [B, C, N, F]
        C = out.size(1)
        out = out.transpose(2, 3).reshape(B, C, N, -1)

        return out


class Dense_TimeDiffPool2d(nn.Module):

    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding):
        super().__init__()

        # TODO: add Normalization
        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kern_size), padding=(0, padding))

        self.re_param = Parameter(Tensor(kern_size, 1))

    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')

    def forward(self, x: Tensor, adj: Tensor):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        x = x.transpose(1, 2)
        out = self.time_conv(x)
        out = out.transpose(1, 2)

        # s: [ N^(l+1), N^l, 1, K ]
        s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)

        # TODO: fully-connect, how to decrease time complexity
        out_adj = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))

        return out, out_adj



