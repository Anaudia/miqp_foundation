import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split

# Data preparation functions
def prepare_edge_index_and_attr(Q):
    n_variables = Q.shape[0]
    edge_index = []
    edge_attr = []
    Q = Q if isinstance(Q, np.ndarray) else Q.toarray()
    for i in range(n_variables):
        for j in range(n_variables):
            if Q[i, j] != 0:
                edge_index.append([i, j])
                edge_attr.append(Q[i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr
    
def create_data_list_obj(solutions, costs, edge_index, edge_attr,
                         variable_types_tensor, variable_lower_bounds, variable_upper_bounds):
    data_list = []
    epsilon = 1e-3
    for x_sol, cost_sol in zip(solutions, costs):
        x_sol_tensor = torch.tensor(x_sol, dtype=torch.float)
        # Variable features: [x_value, variable_type, LB, UB]
        var_features = torch.stack([x_sol_tensor, variable_types_tensor, variable_lower_bounds, variable_upper_bounds], dim=1)
        
        data = Data(x=var_features, edge_index=edge_index, edge_attr=edge_attr)
        data.variable_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        data.binary_mask = variable_types_tensor.bool()
        
        # Targets
        data.y_x = x_sol_tensor
        data.y_cost = torch.tensor([cost_sol], dtype=torch.float)
        
        # Integrality targets for binary variables
        binary_values = x_sol_tensor[data.binary_mask]
        y_integrality = ((binary_values - binary_values.round()).abs() < epsilon).float()
        data.y_integrality = y_integrality

        data_list.append(data)
    return data_list

def normalize_targets(data_list):

    # Normalize y_x (node targets) - exclude binary variables
    y_x_all = []
    for data in data_list:
        # Exclude binary variables from y_x normalization
        continuous_targets = data.y_x[~data.binary_mask]  # Only continuous targets
        y_x_all.append(continuous_targets)

    y_x_all = torch.cat(y_x_all, dim=0)
    mean_y_x = y_x_all.mean()
    std_y_x = y_x_all.std() + 1e-6  # Avoid division by zero

    for data in data_list:
        continuous_targets = data.y_x[~data.binary_mask]
        data.y_x[~data.binary_mask] = (continuous_targets - mean_y_x) / std_y_x

    # Normalize y_cost (graph targets)
    y_cost_all = torch.cat([data.y_cost for data in data_list], dim=0)
    mean_y_cost = y_cost_all.mean()
    std_y_cost = y_cost_all.std() + 1e-6

    for data in data_list:
        data.y_cost = (data.y_cost - mean_y_cost) / std_y_cost

    # Normalize y_constraints (if present)
    mean_y_constraints = None
    std_y_constraints = None
    if hasattr(data_list[0], 'y_constraints'):
        y_constraints_all = torch.cat([data.y_constraints for data in data_list], dim=0)
        mean_y_constraints = y_constraints_all.mean()
        std_y_constraints = y_constraints_all.std() + 1e-6
        for data in data_list:
            data.y_constraints = (data.y_constraints - mean_y_constraints) / std_y_constraints

    return data_list, (mean_y_x, std_y_x), (mean_y_cost, std_y_cost), (mean_y_constraints, std_y_constraints)

def split_data(data_list, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(data_list))
    test_set_size = int(len(data_list) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    train_data = [data_list[i] for i in train_indices]
    test_data = [data_list[i] for i in test_indices]
    return train_data, test_data