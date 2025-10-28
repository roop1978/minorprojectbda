import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class WideComponent(nn.Module):
    """Wide component for memorizing feature interactions"""

    def __init__(self, num_features, embedding_dim=32):
        super(WideComponent, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        self.linear = nn.Linear(num_features, 1)
        self.embeddings = nn.ModuleList([
            nn.Embedding(100, embedding_dim) for _ in range(num_features)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        linear_out = self.linear(x)
        embeddings = []

        for i in range(self.num_features):
            feature_values = torch.clamp(x[:, i].long(), 0, 99)
            emb = self.embeddings[i](feature_values)
            embeddings.append(emb)

        interaction = torch.ones(batch_size, self.embedding_dim, device=x.device)
        for emb in embeddings:
            interaction = interaction * emb

        interaction_out = torch.sum(interaction, dim=1, keepdim=True)
        return linear_out + interaction_out


class DeepComponent(nn.Module):
    """Deep component for learning feature representations"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(DeepComponent, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        return self.network(x)


class SessionGNN(nn.Module):
    """Session-based Graph Neural Network using PyTorch Geometric"""

    def __init__(self, user_dim, job_dim, hidden_dim=64, num_layers=2):
        super(SessionGNN, self).__init__()

        self.user_dim = user_dim
        self.job_dim = job_dim
        self.hidden_dim = hidden_dim

        self.user_encoder = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.job_encoder = nn.Sequential(
            nn.Linear(job_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.output_dim = hidden_dim

    def forward(self, graph, user_features, job_features):
        user_emb = self.user_encoder(user_features)
        job_emb = self.job_encoder(job_features)
        x = torch.cat([user_emb, job_emb], dim=0)

        edge_index = graph.edge_index

        # Ensure the feature matrix matches graph.num_nodes
        num_nodes = getattr(graph, "num_nodes", None)
        if num_nodes is None:
            num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else x.size(0)

        if x.size(0) < num_nodes:
            padded = torch.zeros((num_nodes, x.size(1)), device=x.device, dtype=x.dtype)
            padded[:x.size(0)] = x
            x = padded
        elif x.size(0) > num_nodes:
            x = x[:num_nodes]

        try:
            for conv in self.gnn_layers:
                x = F.relu(conv(x, edge_index))
        except Exception as e:
            # Fallback: skip GCN and use encoded user embeddings
            user_emb_out = user_emb
            user_emb_out = user_emb_out.unsqueeze(0)
            attended, _ = self.attention(user_emb_out, user_emb_out, user_emb_out)
            return attended.squeeze(0)

        user_emb_out = x[:user_emb.size(0)]
        user_emb_out = user_emb_out.unsqueeze(0)
        attended, _ = self.attention(user_emb_out, user_emb_out, user_emb_out)
        return attended.squeeze(0)


class Adversary(nn.Module):
    """Adversarial network for gender bias detection"""

    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=2):
        super(Adversary, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                spectral_norm(nn.Linear(prev_dim, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(spectral_norm(nn.Linear(prev_dim, num_classes)))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FairnessAwareJobRecommender(nn.Module):
    """Main model combining Wide, Deep, GNN, and Adversarial components"""

    def __init__(self, config):
        super(FairnessAwareJobRecommender, self).__init__()

        self.config = config

        self.wide = WideComponent(config['num_wide_features'], config['embedding_dim'])
        self.deep = DeepComponent(config['num_deep_features'], config['deep_hidden_dims'])
        self.session_gnn = SessionGNN(
            config['user_dim'],
            config['job_dim'],
            config['gnn_hidden_dim'],
            config['gnn_num_layers']
        )

        wide_deep_input_dim = 1 + config['deep_hidden_dims'][-1]
        self.adversary = Adversary(
            wide_deep_input_dim,
            config['adversary_hidden_dims']
        )
        fusion_input_dim = (
                config['embedding_dim'] +
                config['deep_hidden_dims'][-1] +
                config['gnn_hidden_dim']
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fairness_layer = nn.Sequential(
            nn.Linear(wide_deep_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )



    def forward(self, batch_data, graph=None):
        wide_features = batch_data['wide_features']
        deep_features = batch_data['deep_features']
        user_features = batch_data['user_features']
        job_features = batch_data['job_features']

        wide_out = self.wide(wide_features)
        deep_out = self.deep(deep_features)

        if graph is not None:
            gnn_out = self.session_gnn(graph, user_features, job_features)
        else:
            gnn_out = torch.zeros(wide_features.size(0), self.session_gnn.output_dim, device=wide_features.device)

        fused_features = torch.cat([
            wide_out.repeat(1, self.config['embedding_dim']),
            deep_out,
            gnn_out
        ], dim=1)

        click_prob = torch.sigmoid(self.fusion(fused_features))

        wide_deep_concat = torch.cat([wide_out, deep_out], dim=1)
        adversary_out = self.adversary(wide_deep_concat)
        fairness_out = self.fairness_layer(wide_deep_concat)

        return {
            'click_prob': click_prob,
            'adversary_out': adversary_out,
            'fairness_out': fairness_out,
            'wide_out': wide_out,
            'deep_out': deep_out,
            'gnn_out': gnn_out
        }


def create_user_job_graph(interactions, user_ids, job_ids):
    """Create a bipartite graph from user-job interactions using PyTorch Geometric"""

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    job_to_idx = {jid: i + len(user_ids) for i, jid in enumerate(job_ids)}  # Offset job node indices

    user_nodes, job_nodes = [], []
    for _, row in interactions.iterrows():
        user_idx = user_to_idx[row['user_id']]
        job_idx = job_to_idx[row['job_id']]
        user_nodes.append(user_idx)
        job_nodes.append(job_idx)

    edge_index = torch.tensor([user_nodes, job_nodes], dtype=torch.long)
    num_nodes = len(user_ids) + len(job_ids)

    graph = Data(edge_index=edge_index, num_nodes=num_nodes)
    return graph, user_to_idx, job_to_idx


def get_model_config():
    """Default model configuration"""
    return {
        'num_wide_features': 10,
        'num_deep_features': 20,
        'user_dim': 15,
        'job_dim': 12,
        'embedding_dim': 32,
        'deep_hidden_dims': [256, 128, 64],
        'gnn_hidden_dim': 64,
        'gnn_num_layers': 2,
        'adversary_hidden_dims': [64, 32],
        'wide_deep_dim': 33
    }
