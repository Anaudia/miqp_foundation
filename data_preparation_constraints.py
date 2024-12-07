import torch
import numpy as np
from torch_geometric.data import Data

def prepare_constraint_edge_data(A, E, n_variables):
    # A: m x n, E: p x n
    m = A.shape[0]
    p = E.shape[0]
    num_constraints = m + p
    edge_index = []
    edge_attr = []
    # Edges from variables to inequality constraints
    for constraint_idx in range(m):
        for variable_idx in range(n_variables):
            coeff = A[constraint_idx, variable_idx]
            if coeff != 0:
                # Edge from variable to constraint
                edge_index.append([variable_idx, n_variables + constraint_idx])
                edge_attr.append(coeff)
    # Edges from variables to equality constraints
    for constraint_idx in range(p):
        for variable_idx in range(n_variables):
            coeff = E[constraint_idx, variable_idx]
            if coeff != 0:
                # Edge from variable to constraint
                edge_index.append([variable_idx, n_variables + m + constraint_idx])
                edge_attr.append(coeff)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr

def create_data_list_feas(solutions, costs, A, E, b_vector, d_vector, edge_index, edge_attr, n_variables,
                          variable_types_tensor, variable_lower_bounds, variable_upper_bounds):
    data_list = []
    m = A.shape[0]
    p = E.shape[0]
    num_constraints = m + p
    A_tensor = torch.tensor(A, dtype=torch.float)
    E_tensor = torch.tensor(E, dtype=torch.float)
    b_torch = torch.tensor(b_vector, dtype=torch.float)
    d_torch = torch.tensor(d_vector, dtype=torch.float)

    epsilon = 1e-3  # Tolerance for checking integrality

    for x_sol, cost_sol in zip(solutions, costs):
        x_sol_tensor = torch.tensor(x_sol, dtype=torch.float)

        # Variable features: [value, var_type, lb, ub]
        var_features = torch.stack([x_sol_tensor, variable_types_tensor, variable_lower_bounds, variable_upper_bounds], dim=1)

        # Constraint features: We store b and d. We pad to match var_features shape.
        b = b_torch.unsqueeze(-1)  # m x 1
        d = d_torch.unsqueeze(-1)  # p x 1
        x_constraints = torch.cat([b, d], dim=0)  # (m+p) x 1
        # Pad constraints to have the same number of feature dimensions
        x_constraints_padded = torch.nn.functional.pad(x_constraints, (0, var_features.shape[1] - x_constraints.shape[1]))
        x_total = torch.cat([var_features, x_constraints_padded], dim=0)

        data = Data(x=x_total, edge_index=edge_index, edge_attr=edge_attr)
        data.variable_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.variable_mask[:n_variables] = True
        data.binary_mask = variable_types_tensor.bool()

        data.y_x = x_sol_tensor  # Unnormalized
        data.y_cost = torch.tensor([cost_sol], dtype=torch.float)

        # Compute constraint violations
        inequality_violations = A_tensor @ x_sol_tensor - b_torch
        equality_violations = E_tensor @ x_sol_tensor - d_torch
        y_constraints = torch.cat([inequality_violations, equality_violations], dim=0)
        data.y_constraints = y_constraints

        # Compute integrality labels for binary variables
        binary_values = x_sol_tensor[data.binary_mask]
        y_integrality = ((binary_values - binary_values.round()).abs() < epsilon).float()
        data.y_integrality = y_integrality

        data_list.append(data)
    return data_list



def normalize_node_features(data_list):
    """
    Normalize node features for continuous variables and constraints separately.
    
    Args:
        data_list: List of graph data objects (torch_geometric.data.Data).

    Returns:
        Normalized data_list, means, and stds for continuous variables and constraints.
    """
    # Initialize storage for global statistics
    means = {"continuous": None, "constraints": None}
    stds = {"continuous": None, "constraints": None}

    # Collect features for continuous variables and constraints
    all_continuous_features = []
    all_constraint_features = []

    # Collect features across all graphs
    for data in data_list:
        # Extract masks
        var_mask = data.variable_mask
        binary_mask = data.binary_mask

        # Continuous variables: Variables not marked as binary
        variable_features = data.x[var_mask]
        continuous_features = variable_features[~binary_mask]  # Exclude binary variables

        # Constraint features: Nodes not marked as variables
        constraint_features = data.x[~var_mask]

        # Append to global collections
        if continuous_features.numel() > 0:  # Ensure non-empty
            all_continuous_features.append(continuous_features)
        if constraint_features.numel() > 0:
            all_constraint_features.append(constraint_features)

    if len(all_continuous_features) > 0:
        all_continuous_features = torch.cat(all_continuous_features, dim=0)
        means["continuous"] = all_continuous_features.mean(dim=0)
        stds["continuous"] = all_continuous_features.std(dim=0) + 1e-6
    else:
        # If no continuous features, create dummy means/stds
        means["continuous"] = torch.zeros(4)  # match feature dimension
        stds["continuous"] = torch.ones(4)

    if len(all_constraint_features) > 0:
        all_constraint_features = torch.cat(all_constraint_features, dim=0)
        means["constraints"] = all_constraint_features.mean(dim=0)
        stds["constraints"] = all_constraint_features.std(dim=0) + 1e-6
    else:
        # If no constraints, create dummy means/stds
        means["constraints"] = torch.zeros(4)
        stds["constraints"] = torch.ones(4)

    # Apply normalization to each graph
    for data in data_list:
        # Continuous variables
        variable_features = data.x[data.variable_mask]
        continuous_mask = ~data.binary_mask
        if continuous_mask.any():
            continuous_values = variable_features[continuous_mask]
            variable_features[continuous_mask] = (
                continuous_values - means["continuous"]
            ) / stds["continuous"]
            data.x[data.variable_mask] = variable_features

        # Constraints
        constraint_mask = ~data.variable_mask
        if constraint_mask.any():
            data.x[constraint_mask] = (
                data.x[constraint_mask] - means["constraints"]
            ) / stds["constraints"]

    return data_list, means, stds
