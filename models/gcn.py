import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_node_features: int, num_classes: int, num_layers=3):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        num_input_features = num_node_features
        for i in range(num_layers):
            conv = GCNConv(num_input_features, hidden_channels)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
            num_input_features = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x
