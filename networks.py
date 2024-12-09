import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class JointGNN(nn.Module):
    def __init__(self, hidden_channels_obj, hidden_channels_cons, decoder_hidden_channels):
        super(JointGNN, self).__init__()
        # Encoders
        self.encoder_obj = GNNModelObj(hidden_channels_obj)
        self.encoder_cons = GNNModelConstraints(hidden_channels_cons)

        concat_dim = hidden_channels_obj + hidden_channels_cons

        # Decoder for x reconstruction
        self.decoder_x = nn.Sequential(
            nn.Linear(concat_dim, decoder_hidden_channels),
            nn.ReLU(),
            nn.Linear(decoder_hidden_channels, 1)
        )

        # Decoder for cost prediction
        self.decoder_cost = nn.Sequential(
            nn.Linear(concat_dim, decoder_hidden_channels),
            nn.ReLU(),
            nn.Linear(decoder_hidden_channels, 1)
        )

        # Decoder for constraint violation prediction
        self.decoder_constraints = nn.Sequential(
            nn.Linear(hidden_channels_cons, decoder_hidden_channels),
            nn.ReLU(),
            nn.Linear(decoder_hidden_channels, 1)
        )

        # Decoder for integrality prediction
        self.decoder_integrality = nn.Sequential(
            nn.Linear(concat_dim, decoder_hidden_channels),
            nn.ReLU(),
            nn.Linear(decoder_hidden_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, data_obj, data_feas):
        x_obj = self.encoder_obj(data_obj)
        x_cons_var, x_cons_constraints = self.encoder_cons(data_feas)

        x_obj_var = x_obj[data_obj.variable_mask]
        x_var = torch.cat([x_obj_var, x_cons_var], dim=1)

        x_hat = self.decoder_x(x_var).squeeze()

        batch = data_obj.batch[data_obj.variable_mask]
        x_var_pooled = global_mean_pool(x_var, batch)
        predicted_cost = self.decoder_cost(x_var_pooled).squeeze()

        predicted_constraints = self.decoder_constraints(x_cons_constraints).squeeze()

        integrality_scores = self.decoder_integrality(x_var).squeeze()
        binary_mask = data_obj.binary_mask[data_obj.variable_mask]
        predicted_integrality = integrality_scores[binary_mask]

        return x_hat, predicted_cost, predicted_constraints, predicted_integrality


class GNNModelObj(nn.Module):
    def __init__(self, hidden_channels):
        super(GNNModelObj, self).__init__()
        self.conv1 = GCNConv(4, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        return x


class GNNModelConstraints(nn.Module):
    def __init__(self, hidden_channels):
        super(GNNModelConstraints, self).__init__()
        self.conv1 = GCNConv(4, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x_var = x[data.variable_mask]
        x_constraints = x[~data.variable_mask]
        return x_var, x_constraints
