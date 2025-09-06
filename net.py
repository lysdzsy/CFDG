from math import ceil
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from layer import *
import networkx as nx
import matplotlib.pyplot as plt

class GNNStack(nn.Module):
    """ The stack layers of GNN.

    """


    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size,
                 in_dim, hidden_dim, out_dim, attention_dim, rho,
                 seq_len, num_nodes, num_classes, dropout=0.5, activation=nn.ReLU(),regularizations=None):

        super().__init__()
        self.attention_dim = attention_dim
        self.rho = rho
        self.emb_dims = [256,256]
        self.attn_dim = 64
        self.lga_learners = nn.ModuleList([
            LGALearner(self.emb_dims[i], self.attn_dim) for i in range(len(self.emb_dims))
        ])
        self.iv_module = InstrumentalVariableModule(
            x_dim=256,
            y_dim=num_classes,
            z_dim=256,
            hidden_dim=256
        )
        self.num_classes=num_classes
        self.regularizations = regularizations if regularizations is not None else ["feature_smoothing", "degree",
                                                                                    "sparse"]
        # TODO: Sparsity Analysis
        k_neighs = self.num_nodes = num_nodes
        self.num_graphs = groups

        self.num_feats = seq_len
        if seq_len % groups:
            self.num_feats += ( groups - seq_len % groups )
        self.g_constr = multi_shallow_embedding(num_nodes, k_neighs, self.num_graphs)

        gnn_model, heads = self.build_gnn_model(gnn_model_type)

        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'
        paddings = [ (k - 1) // 2 for k in kern_size ]

        self.tconvs = nn.ModuleList(
            [nn.Conv2d(1, in_dim, (1, kern_size[0]), padding=(0, paddings[0]))] +
            [nn.Conv2d(heads * in_dim, hidden_dim, (1, kern_size[layer+1]), padding=(0, paddings[layer+1])) for layer in range(num_layers - 2)] +
            [nn.Conv2d(heads * hidden_dim, out_dim, (1, kern_size[-1]), padding=(0, paddings[-1]))]
        )

        self.gconvs = nn.ModuleList(
            [gnn_model(in_dim, heads * in_dim, groups)] +
            [gnn_model(hidden_dim, heads * hidden_dim, groups) for _ in range(num_layers - 2)] +
            [gnn_model(out_dim, heads * out_dim, groups)]
        )

        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(heads * in_dim)] +
            [nn.BatchNorm2d(heads * hidden_dim) for _ in range(num_layers - 2)] +
            [nn.BatchNorm2d(heads * out_dim)]
        )

        self.left_num_nodes = []
        for layer in range(num_layers + 1):
            left_node = round( num_nodes * (1 - (pool_ratio*layer)) )
            if left_node > 0:
                self.left_num_nodes.append(left_node)
            else:
                self.left_num_nodes.append(1)

        self.diffpool = nn.ModuleList(
            [Dense_TimeDiffPool2d(self.left_num_nodes[layer], self.left_num_nodes[layer+1], kern_size[layer], paddings[layer]) for layer in range(num_layers - 1)] +
            [Dense_TimeDiffPool2d(self.left_num_nodes[-2], self.left_num_nodes[-1], kern_size[-1], paddings[-1])]
        )

        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(heads * out_dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for tconv, gconv, bn, pool in zip(self.tconvs, self.gconvs, self.bns, self.diffpool):
            tconv.reset_parameters()
            gconv.reset_parameters()
            bn.reset_parameters()
            pool.reset_parameters()

        self.linear.reset_parameters()


    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1
        if model_type == 'dyGSA2d':
            return DenseGraphSAGEConv2d,1



    def forward(self, inputs: Tensor, label):
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            x = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = inputs

        adj = self.g_constr(x.device)
        B_graph, N = adj.shape[0], adj.shape[1]
        B = x.size(0)
        #print(x.shape)
        x = self.tconvs[0](x)
        x = self.gconvs[0](x, adj)
        adj = self.update_adj_with_attention(x, adj, self.lga_learners[0])

        x = self.tconvs[1](x)
        adj_updated = self.update_adj_with_attention(x, adj, self.lga_learners[1])

        z_causal_emb = self.run_graph_path(x, adj_updated)
        z_noncausal_emb= self.run_graph_path(x, 1.0 - adj_updated)

        y_onehot = F.one_hot(label, num_classes=self.num_classes).float()
        mask = torch.rand(y_onehot.size(0), device=y_onehot.device) >0.3
        y_onehot = y_onehot * mask.unsqueeze(1)
        z, mu, logvar = self.iv_module(z_noncausal_emb, y_onehot)

        out = self.linear(z_causal_emb)
        cf_emb_out = self.linear(z)

        z_noncausal_perm1 = self.random_permute(z)
        z_noncausal_perm_out1 = self.linear(z_noncausal_perm1)

        z_noncausal_swap1 = self.swap_mean_var(z, z_noncausal_perm1)
        z_noncausal_swap_out1 = self.linear(z_noncausal_swap1)

        z_noncausal_perm2 = self.subspace_perturb(z)
        z_noncausal_perm_out2 = self.linear(z_noncausal_perm2)

        z_noncausal_swap2 = self.local_block_mixup(z)
        z_noncausal_swap_out2 = self.linear(z_noncausal_swap2)

        return out, z_noncausal_perm_out1, z_noncausal_swap_out1,z_noncausal_perm_out2, z_noncausal_swap_out2, z_causal_emb,z_noncausal_emb, cf_emb_out, z_noncausal_perm1, z_noncausal_swap1,z_noncausal_perm2, z_noncausal_swap2

    def update_adj_with_attention(self, node_feature: Tensor, adj: Tensor, lga_learner: nn.Module) -> Tensor:
        B_graph, N = adj.shape[0], adj.shape[1]#[4,6,6]    [4,4,4]
        node_emb = node_feature.mean(dim=-1).permute(0, 2, 1).contiguous().view(B_graph * N, -1)

        adj_updated = torch.zeros_like(adj)

        for i in range(B_graph):
            sub_adj = adj[i]
            sub_edge_index = sub_adj.nonzero(as_tuple=False).T  # [2, E]
            sub_node_emb = node_emb[i * N:(i + 1) * N]

            raw_logits = lga_learner(sub_node_emb, sub_edge_index)
            raw_logits = torch.sigmoid(raw_logits)

            edge_score_dict = {}
            for j in range(sub_edge_index.size(1)):
                u, v = sub_edge_index[0, j].item(), sub_edge_index[1, j].item()
                key = (u, v)
                edge_score_dict.setdefault(key, []).append(raw_logits[j])

            new_logits, new_index = [], [[], []]
            for (u, v), scores in edge_score_dict.items():
                avg_score = torch.stack(scores).mean()
                new_logits.extend([avg_score, avg_score])
                new_index[0].extend([u, v])
                new_index[1].extend([v, u])

            sub_edge_index = torch.tensor(new_index, device=adj.device)
            sub_edge_logits = torch.stack(new_logits)
            mask = self.gumbel_sigmoid_sample(sub_edge_logits, hard=False)

            for j in range(sub_edge_index.size(1)):
                u, v = sub_edge_index[0, j], sub_edge_index[1, j]
                adj_updated[i, u, v] = mask[j]

        return adj_updated

    def gumbel_sigmoid_sample(self, logits, temperature=0.5, hard=False):
        eps = 1e-20
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y = torch.sigmoid((logits + gumbel_noise) / temperature)
        if hard:
            y_hard = (y > 0.5).float()
            return (y_hard - y).detach() + y
        return y

    def run_graph_path(self, x, adj):
        for tconv, gconv, bn, pool in zip(self.tconvs[1:], self.gconvs[1:], self.bns[1:], self.diffpool):
            #print(x.shape)#[16, 64, 6, 1024])
            x2 = tconv(x)
            #print(x.shape)#([16, 64, 5, 1024])
            x2 = gconv(x2, adj)
            x, adj = pool(x2, adj)
            x = self.activation(bn(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        return out


    def random_permute(self, z):
        idx = torch.randperm(z.size(0))
        return z[idx]

    def swap_mean_var(self, z, z_permuted):
        mean = z.mean(dim=0, keepdim=True)
        std = z.std(dim=0, keepdim=True) + 1e-6
        mean_p = z_permuted.mean(dim=0, keepdim=True)
        std_p = z_permuted.std(dim=0, keepdim=True)
        return (z - mean) / std * std_p + mean_p

    def subspace_perturb(self, z, perturb_strength=0.05):
        B, D = z.shape
        W = torch.randn(D, D, device=z.device)
        Q, _ = torch.linalg.qr(W)
        z_rotated = torch.matmul(z, Q)
        return (1 - perturb_strength) * z + perturb_strength * z_rotated

    def local_block_mixup(self, z, block_ratio=0.2):
        B, D = z.shape
        if isinstance(D, torch.Tensor):
            D = D.item()
        z_perm = z[torch.randperm(B)]
        num_blocks = max(1, int(D * block_ratio))
        idx = torch.randperm(D)[:num_blocks]
        z_mixed = z.clone()
        z_mixed[:, idx] = (z[:, idx] + z_perm[:, idx]) / 2
        return z_mixed





