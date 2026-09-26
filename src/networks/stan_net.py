import torch
import torch.nn as nn
import torch.nn.functional as F

class STANNet(nn.Module):
    """
    Spatial-Temporal Attention Network (STAN-NIDS) for IoT Network Intrusion Detection.
    
    Architecture:
    1. Spatial 1D-Conv Feature Projection: Extracts local feature interactions per packet.
    2. Bidirectional GRU (Temporal Layer): Captures sequential flow dynamics across the 10-packet window.
    3. Multi-Head Self-Attention: Learns long-range dependencies across all packets, removing single-packet bias.
    4. Multi-Scale Attentive Pooling: Combines attention-weighted sequence representation with max pooling.
    5. Latent Feature Bottleneck (200-dim): Compatible with existing heads, replay memory, and XAI tools.
    """

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()
        num_pkts = kwargs.get('num_pkts', 10)
        num_fields = kwargs.get('num_fields', 4)
        out_features_size = 200
        d_model = 64
        
        # 1. Spatial feature projection
        self.proj = nn.Linear(num_fields, d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        
        # 2. Bidirectional Temporal Modeling
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # 3. Multi-Head Self-Attention (128-dim = 64 * 2 from BiGRU)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(128)
        
        # 4. Attentive sequence pooling weight
        self.pool_weight = nn.Linear(128, 1)
        
        # 5. Latent representation (Attention-pooled + Max-pooled = 256 -> 200)
        self.fc1 = nn.Linear(128 * 2, out_features_size)
        self.bn_out = nn.BatchNorm1d(out_features_size)
        self.dropout = nn.Dropout(0.2)
        
        # 6. Classifier Head
        self.fc = nn.Linear(out_features_size, num_classes)
        self.head_var = 'fc'

    def extract_features(self, x):
        """
        Extracts 200-dimensional latent feature vector for LLL_Net wrapper, memory, and XAI.
        Input x shape: (B, 1, num_pkts, num_fields) or (B, num_pkts, num_fields)
        Output shape: (B, 200)
        """
        if x.dim() == 4:
            x = x.squeeze(1) # Convert (B, 1, 10, 4) -> (B, 10, 4)
        
        # Project fields per packet: (B, 10, 4) -> (B, 10, 64)
        x_proj = F.relu(self.proj(x))
        
        # 1D Temporal convolution across packet sequence: (B, 64, 10)
        x_conv = x_proj.transpose(1, 2)
        x_conv = F.relu(self.bn1(self.conv1(x_conv)))
        x_conv = x_conv.transpose(1, 2) # (B, 10, 64)
        
        # Bi-directional GRU across packet sequence: (B, 10, 128)
        gru_out, _ = self.gru(x_conv)
        
        # Multi-Head Self-Attention over packets
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        norm_out = self.norm(gru_out + attn_out) # (B, 10, 128)
        
        # Multi-scale pooling: Attention-weighted mean + Max pooling
        attn_weights = F.softmax(self.pool_weight(norm_out), dim=1) # (B, 10, 1)
        attn_pooled = torch.sum(norm_out * attn_weights, dim=1)     # (B, 128)
        max_pooled, _ = torch.max(norm_out, dim=1)                  # (B, 128)
        
        # Combine representations: (B, 256) -> (B, 200)
        combined = torch.cat([attn_pooled, max_pooled], dim=1)
        out = self.dropout(self.bn_out(F.relu(self.fc1(combined))))
        return out

    def forward(self, x):
        features = F.relu(self.extract_features(x))
        return self.fc(features)
