import torch
import numpy as np
from torch_geometric.data import Data

def prepare_constraint_edge_data(A, E, n, m, p):
    """
    Prepares edge indices and attributes from matrices A and E for constraint graphs.

    Args:
        A (np.ndarray): Inequality constraint matrix (m x n).
        E (np.ndarray): Equality constraint matrix (p x n).
        n (int): Number of variables.
        m (int): Number of inequality constraints.
        p (int): Number of equality constraints.

    Returns:
        edge_index (torch.Tensor): Edge indices for the graph.
        edge_attr (torch.Tensor): Edge attributes (weights) for the graph.
    """
    # Get non-zero elements in A
    A_row_indices, A_col_indices = np.nonzero(A)
    A_edge_weights = A[A_row_indices, A_col_indices]

    # Get non-zero elements in E
    E_row_indices, E_col_indices = np.nonzero(E)
    E_edge_weights = E[E_row_indices, E_col_indices]

    # Edges from variable nodes to inequality constraint nodes
    A_edge_index = torch.tensor(
        [A_col_indices, A_row_indices + n], dtype=torch.long
    )
    A_edge_attr = torch.tensor(A_edge_weights, dtype=torch.float)

    # Edges from variable nodes to equality constraint nodes
    E_edge_index = torch.tensor(
        [E_col_indices, E_row_indices + n + m], dtype=torch.long
    )
    E_edge_attr = torch.tensor(E_edge_weights, dtype=torch.float)

    # Combine edge indices and attributes
    edge_index = torch.cat([A_edge_index, E_edge_index], dim=1)
    edge_attr = torch.cat([A_edge_attr, E_edge_attr], dim=0)

    return edge_index, edge_attr

def create_constraint_data_list(feasible_solutions, A, E, b_vector, d_vector, edge_index, edge_attr, n, m, p):
    """
    Creates a list of Data objects for constraint prediction.

    Args:
        feasible_solutions (list): List of feasible solutions (numpy arrays).
        A (np.ndarray): Inequality constraint matrix (m x n).
        E (np.ndarray): Equality constraint matrix (p x n).
        b_vector (np.ndarray): Right-hand side vector for inequalities (length m).
        d_vector (np.ndarray): Right-hand side vector for equalities (length p).
        edge_index (torch.Tensor): Edge indices for the graph.
        edge_attr (torch.Tensor): Edge attributes (weights) for the graph.
        n (int): Number of variables.
        m (int): Number of inequality constraints.
        p (int): Number of equality constraints.

    Returns:
        data_list (list): List of Data objects.
    """
    data_list = []
    for x in feasible_solutions:
        # Node features
        variable_node_features = torch.tensor(x, dtype=torch.float).unsqueeze(1)
        inequality_constraint_node_features = torch.tensor(b_vector, dtype=torch.float).unsqueeze(1)
        equality_constraint_node_features = torch.tensor(d_vector, dtype=torch.float).unsqueeze(1)
        node_features = torch.cat([
            variable_node_features,
            inequality_constraint_node_features,
            equality_constraint_node_features
        ], dim=0)

        # Target y is [Ax - b; Ex - d], concatenate both
        inequality_violation = A @ x - b_vector  # Size m
        equality_violation = E @ x - d_vector    # Size p
        y = np.concatenate([inequality_violation, equality_violation])
        y = torch.tensor(y, dtype=torch.float)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
        data.num_nodes = n + m + p  # Explicitly set num_nodes
        data_list.append(data)
    return data_list

def normalize_constraint_variable_features(data_list, n):
    """
    Normalizes variable node features for constraint data.

    Args:
        data_list (list): List of Data objects.
        n (int): Number of variable nodes.

    Returns:
        tuple: (data_list, mean_var, std_var)
            data_list: List of Data objects with normalized variable node features.
            mean_var: Mean of the variable node features.
            std_var: Standard deviation of the variable node features.
    """
    all_variable_node_features = torch.cat([data.x[:n] for data in data_list], dim=0)
    mean_var = all_variable_node_features.mean()
    std_var = all_variable_node_features.std()
    for data in data_list:
        data.x[:n] = (data.x[:n] - mean_var) / std_var
    return data_list, mean_var, std_var

def normalize_constraint_targets(data_list):
    """
    Normalizes target y for constraint data.

    Args:
        data_list (list): List of Data objects.

    Returns:
        tuple: (data_list, target_mean, target_std)
            data_list: List of Data objects with normalized targets.
            target_mean: Mean of the targets.
            target_std: Standard deviation of the targets.
    """
    all_targets = torch.cat([data.y for data in data_list], dim=0)
    target_mean = all_targets.mean()
    target_std = all_targets.std()
    for data in data_list:
        data.y = (data.y - target_mean) / target_std
    return data_list, target_mean, target_std