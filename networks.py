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

        self.hidden_channels_obj = hidden_channels_obj
        self.hidden_channels_cons = hidden_channels_cons
        concat_dim = hidden_channels_obj + hidden_channels_cons

        # For the VAE: transform the combined embeddings into a latent distribution
        self.fc_mu = nn.Linear(concat_dim, concat_dim)
        self.fc_logvar = nn.Linear(concat_dim, concat_dim)

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
            nn.Linear(concat_dim, decoder_hidden_channels),
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
        # Encode object nodes
        z_obj_var = self.encoder_obj(data_obj)  # (N_obj_nodes, hidden_channels_obj)
        # Encode feasibility nodes (variables and constraints)
        z_cons_var, z_cons_constraints = self.encoder_cons(data_feas) 

        # Concatenate variable embeddings
        z_var = torch.cat([z_obj_var, z_cons_var], dim=1) # shape: (num_var_nodes, obj_dim + cons_dim)

        # We need to pad constraints to match dimension for concatenation
        z_cons_constraints_padded = F.pad(z_cons_constraints, (0, z_var.size(1) - z_cons_constraints.size(1)))

        # Concatenate variable and constraint embeddings along node dimension
        # shape: (num_var_nodes + num_cons_nodes, concat_dim)
        z_shared = torch.cat([z_var, z_cons_constraints_padded], dim=0)

        # Now, create latent distribution parameters for VAE
        z_mu = self.fc_mu(z_shared)       # (num_nodes_total, concat_dim)
        z_logvar = self.fc_logvar(z_shared)

        # Sample latent vector z from q(z|x) = N(z_mu, exp(z_logvar))
        # Reparameterization trick
        eps = torch.randn_like(z_mu)
        z = z_mu + torch.exp(0.5 * z_logvar) * eps

        # Now decode from z
        num_var_nodes = z_var.size(0)
        z_shared_var = z[:num_var_nodes]            # variable embeddings
        z_shared_constraints = z[num_var_nodes:]    # constraint embeddings

        batch = data_feas.batch[data_feas.variable_mask]

        # x reconstruction
        x_hat = self.decoder_x(z_shared_var).squeeze()  # (num_var_nodes,)

        # Cost prediction
        z_var_pooled = global_mean_pool(z_shared_var, batch)
        predicted_cost = self.decoder_cost(z_var_pooled).squeeze()

        # Constraint violation prediction
        predicted_constraints = self.decoder_constraints(z_shared_constraints).squeeze()

        # Integrality prediction
        integrality_scores = self.decoder_integrality(z_shared_var).squeeze()
        binary_mask = data_obj.binary_mask[data_obj.variable_mask]
        predicted_integrality = integrality_scores[binary_mask]

        return x_hat, predicted_cost, predicted_constraints, predicted_integrality, z_mu, z_logvar


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
