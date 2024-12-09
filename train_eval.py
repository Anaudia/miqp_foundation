import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np

def train(train_loader, model, optimizer, criterion_mse, criterion_bce, device='cuda'):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_cost_loss = 0
    total_constraint_loss = 0
    total_integrality_loss = 0

    for data_obj_batch, data_feas_batch in train_loader:
        data_obj_batch = data_obj_batch.to(device)
        data_feas_batch = data_feas_batch.to(device)

        optimizer.zero_grad()
        x_hat, predicted_cost, predicted_constraints, predicted_integrality = model(data_obj_batch, data_feas_batch)

        y_x = data_obj_batch.y_x
        y_cost = data_obj_batch.y_cost.squeeze()
        y_constraints = data_feas_batch.y_constraints
        y_integrality = data_obj_batch.y_integrality.to(device)

        recon_loss = criterion_mse(x_hat, y_x)
        cost_loss = criterion_mse(predicted_cost, y_cost)
        constraint_loss = criterion_mse(predicted_constraints, y_constraints)
        integrality_loss = criterion_bce(predicted_integrality, y_integrality)

        loss = recon_loss + cost_loss + constraint_loss + integrality_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data_obj_batch.num_graphs
        total_recon_loss += recon_loss.item() * data_obj_batch.num_graphs
        total_cost_loss += cost_loss.item() * data_obj_batch.num_graphs
        total_constraint_loss += constraint_loss.item() * data_obj_batch.num_graphs
        total_integrality_loss += integrality_loss.item() * data_obj_batch.num_graphs

    n = len(train_loader.dataset)
    return (total_loss/n, total_recon_loss/n, total_cost_loss/n, total_constraint_loss/n, total_integrality_loss/n)


def test(loader, model, criterion_mse, criterion_bce, device='cuda'):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_cost_loss = 0
    total_constraint_loss = 0
    total_integrality_loss = 0

    with torch.no_grad():
        for data_obj_batch, data_feas_batch in loader:
            data_obj_batch = data_obj_batch.to(device)
            data_feas_batch = data_feas_batch.to(device)

            x_hat, predicted_cost, predicted_constraints, predicted_integrality = model(data_obj_batch, data_feas_batch)

            y_x = data_obj_batch.y_x
            y_cost = data_obj_batch.y_cost.squeeze()
            y_constraints = data_feas_batch.y_constraints
            y_integrality = data_obj_batch.y_integrality.to(device)

            recon_loss = criterion_mse(x_hat, y_x)
            cost_loss = criterion_mse(predicted_cost, y_cost)
            constraint_loss = criterion_mse(predicted_constraints, y_constraints)
            integrality_loss = criterion_bce(predicted_integrality, y_integrality)

            loss = recon_loss + cost_loss + constraint_loss + integrality_loss

            total_loss += loss.item() * data_obj_batch.num_graphs
            total_recon_loss += recon_loss.item() * data_obj_batch.num_graphs
            total_cost_loss += cost_loss.item() * data_obj_batch.num_graphs
            total_constraint_loss += constraint_loss.item() * data_obj_batch.num_graphs
            total_integrality_loss += integrality_loss.item() * data_obj_batch.num_graphs

    n = len(loader.dataset)
    return (total_loss/n, total_recon_loss/n, total_cost_loss/n, total_constraint_loss/n, total_integrality_loss/n)
