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

        # **Changed**: Decoder for constraint violation prediction now also uses concat_dim
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
        z_obj_var = self.encoder_obj(data_obj)  # (N_obj_nodes, hidden_channels_obj)
        z_cons_var, z_cons_constraints = self.encoder_cons(data_feas)  
            
        # Concatenate obj and cons var features along the feature dimension
        z_var = torch.cat([z_obj_var, z_cons_var], dim=1)  # (num_var_nodes, hidden_channels_obj + hidden_channels_cons)
    
        # Pad constraints to match dimension
        z_cons_constraints_padded = F.pad(z_cons_constraints, (0, z_var.size(1) - z_cons_constraints.size(1)))
    
        # Concatenate variables and constraints: (num_var_nodes + num_cons_nodes, concat_dim)
        z_shared = torch.cat([z_var, z_cons_constraints_padded], dim=0)  
    
        # Split back into variable and constraint parts
        num_var_nodes = z_var.size(0)
        z_shared_var = z_shared[:num_var_nodes]         # (num_var_nodes, concat_dim)
        z_shared_constraints = z_shared[num_var_nodes:] # (num_cons_nodes, concat_dim)
    
        # Use data_feas.batch since z_cons_var came from data_feas
        batch = data_feas.batch[data_feas.variable_mask]
    
        # x reconstruction
        x_hat = self.decoder_x(z_shared_var).squeeze()
            
        # Cost prediction - use the same batch indexing from data_feas
        z_var_pooled = global_mean_pool(z_shared_var, batch)
        predicted_cost = self.decoder_cost(z_var_pooled).squeeze()
    
        # Constraint violation prediction
        predicted_constraints = self.decoder_constraints(z_shared_constraints).squeeze()
    
        # Integrality prediction
        integrality_scores = self.decoder_integrality(z_shared_var).squeeze()
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
