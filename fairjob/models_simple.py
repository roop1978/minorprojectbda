import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np


class WideComponent(nn.Module):
    """Wide component for memorizing feature interactions"""
    
    def __init__(self, num_features, embedding_dim=32):
        super(WideComponent, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # Linear layer for wide features
        self.linear = nn.Linear(num_features, 1)
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(100, embedding_dim) for _ in range(num_features)
        ])
        
    def forward(self, x):
        # x shape: (batch_size, num_features)
        batch_size = x.size(0)
        
        # Linear transformation
        linear_out = self.linear(x)
        
        # Embedding lookup and interaction
        embeddings = []
        for i in range(self.num_features):
            # Convert continuous features to categorical for embedding
            feature_values = torch.clamp(x[:, i].long(), 0, 99)
            emb = self.embeddings[i](feature_values)
            embeddings.append(emb)
        
        # Element-wise product of embeddings (feature interaction)
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


class SimpleSessionGNN(nn.Module):
    """Simplified session-based component without DGL"""
    
    def __init__(self, user_dim, job_dim, hidden_dim=64, num_layers=2):
        super(SimpleSessionGNN, self).__init__()
        
        self.user_dim = user_dim
        self.job_dim = job_dim
        self.hidden_dim = hidden_dim
        
        # User and job encoders
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
        
        # Simple attention mechanism for session aggregation
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        self.output_dim = hidden_dim
        
    def forward(self, user_features, job_features):
        # Encode user and job features
        user_embeddings = self.user_encoder(user_features)
        job_embeddings = self.job_encoder(job_features)
        
        # Simple interaction modeling (without graph)
        interaction = user_embeddings * job_embeddings
        
        # Session-level attention
        user_embeddings_att = user_embeddings.unsqueeze(0)  # (1, batch_size, hidden_dim)
        attended_features, _ = self.attention(
            user_embeddings_att, user_embeddings_att, user_embeddings_att
        )
        attended_features = attended_features.squeeze(0)  # (batch_size, hidden_dim)
        
        return attended_features + interaction


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


class SimpleFairnessAwareJobRecommender(nn.Module):
    """Simplified model combining Wide, Deep, Session, and Adversarial components"""
    
    def __init__(self, config):
        super(SimpleFairnessAwareJobRecommender, self).__init__()
        
        self.config = config
        
        # Components
        self.wide = WideComponent(config['num_wide_features'], config['embedding_dim'])
        self.deep = DeepComponent(config['num_deep_features'], config['deep_hidden_dims'])
        self.session_gnn = SimpleSessionGNN(
            config['user_dim'], 
            config['job_dim'], 
            config['gnn_hidden_dim'],
            config['gnn_num_layers']
        )
        self.adversary = Adversary(
            config['wide_deep_dim'], 
            config['adversary_hidden_dims']
        )
        
        # Fusion layer
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
        
        # Fairness regularization
        self.fairness_layer = nn.Sequential(
            nn.Linear(config['wide_deep_dim'], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, batch_data, graph=None):
        wide_features = batch_data['wide_features']
        deep_features = batch_data['deep_features']
        user_features = batch_data['user_features']
        job_features = batch_data['job_features']
        
        # Wide component
        wide_out = self.wide(wide_features)
        
        # Deep component
        deep_out = self.deep(deep_features)
        
        # Session GNN component
        gnn_out = self.session_gnn(user_features, job_features)
        
        # Fuse all components
        fused_features = torch.cat([
            wide_out.repeat(1, self.config['embedding_dim']),
            deep_out, 
            gnn_out
        ], dim=1)
        
        # Main prediction
        click_prob = torch.sigmoid(self.fusion(fused_features))
        
        # Adversarial prediction (for bias detection)
        wide_deep_concat = torch.cat([wide_out, deep_out], dim=1)
        adversary_out = self.adversary(wide_deep_concat)
        
        # Fairness regularization
        fairness_out = self.fairness_layer(wide_deep_concat)
        
        return {
            'click_prob': click_prob,
            'adversary_out': adversary_out,
            'fairness_out': fairness_out,
            'wide_out': wide_out,
            'deep_out': deep_out,
            'gnn_out': gnn_out
        }


def get_model_config():
    """Get default model configuration"""
    return {
        'num_wide_features': 10,  # matches wide_features in create_batch_data
        'num_deep_features': 20,  # matches deep_features in create_batch_data
        'user_dim': 15,  # matches user_features in create_batch_data
        'job_dim': 12,   # matches job_features in create_batch_data
        'embedding_dim': 32,
        'deep_hidden_dims': [256, 128, 64],
        'gnn_hidden_dim': 64,
        'gnn_num_layers': 2,
        'adversary_hidden_dims': [64, 32],
        'wide_deep_dim': 65  # 1 + 64 (wide output + deep output)
    }
