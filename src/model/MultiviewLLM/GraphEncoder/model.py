from torch import nn
import torch
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
from src.model.MultiviewLLM.GraphEncoder.layer import make_gnn_layer


class Encoder(nn.Module):
    def __init__(self, input_dim, edge_dim, layer_mode, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gnn_layer(layer_mode, input_dim, hidden_dim, edge_dim))
            else:
                self.layers.append(make_gnn_layer(layer_mode, hidden_dim, hidden_dim, edge_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index, edge_attr=edge_attr)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class EncoderLearner(torch.nn.Module):
    def __init__(self, mcc_num, edge_dim, layer_mode, hidden_dim, num_layers, augmentor, mcc_embed, semantic_initial):
        super(EncoderLearner, self).__init__()
        self.semantic_embed = nn.Embedding.from_pretrained(mcc_embed, freeze=True)
        if not semantic_initial:
            self.mcc_embed = nn.Embedding(mcc_num, hidden_dim)
        else:
            self.lin_mcc = nn.Linear(mcc_embed.size(1), hidden_dim)
        self.encoder = Encoder(hidden_dim, edge_dim, layer_mode, hidden_dim, num_layers)
        self.augmentor = augmentor

        # Define projector for graph-level and node-level representations
        project_dim = hidden_dim * num_layers
        # # Graph-level projector
        self.graph_projector = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))
        # # Node-level projector
        self.node_projector = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))
        # # Mcc-level projector
        self.mcc_projector = torch.nn.Sequential(
            nn.Linear(mcc_embed.size(1), project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        semantic_embed = self.semantic_embed(x.squeeze())
        if not hasattr(self, 'lin_mcc'):
            x = self.mcc_embed(x.squeeze())
        else:
            x = self.lin_mcc(semantic_embed)
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_attr)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_attr)
        z, g = self.encoder(x, edge_index, edge_attr, batch)
        z1, g1 = self.encoder(x1, edge_index1, edge_weight1, batch)
        z2, g2 = self.encoder(x2, edge_index2, edge_weight2, batch)
        return z, g, z1, z2, g1, g2, semantic_embed

    def project_graph(self, g):
        return self.graph_projector(g)

    def project_node(self, z):
        return self.node_projector(z)

    def project_mcc(self, mcc):
        return self.mcc_projector(mcc)