import torch
import torch.nn as nn
import torch.nn.functional as F

def kld_loss(mu, logvar):
    # KL divergence between q(z|x) and p(z) = N(0,I):
    # D_KL(q(z|x)||p(z)) = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def train(train_loader, model, optimizer, criterion_mse, criterion_bce, device='cuda', beta=1e-3):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_cost_loss = 0
    total_constraint_loss = 0
    total_integrality_loss = 0
    total_kld_loss = 0

    for data_obj_batch, data_feas_batch in train_loader:
        data_obj_batch = data_obj_batch.to(device)
        data_feas_batch = data_feas_batch.to(device)

        optimizer.zero_grad()
        x_hat, predicted_cost, predicted_constraints, predicted_integrality, z_mu, z_logvar = model(data_obj_batch, data_feas_batch)

        y_x = data_obj_batch.y_x
        y_cost = data_obj_batch.y_cost.squeeze()
        y_constraints = data_feas_batch.y_constraints
        y_integrality = data_obj_batch.y_integrality.to(device)

        recon_loss = criterion_mse(x_hat, y_x)
        cost_loss = criterion_mse(predicted_cost, y_cost)
        constraint_loss = criterion_mse(predicted_constraints, y_constraints)
        integrality_loss = criterion_bce(predicted_integrality, y_integrality)

        # KL divergence
        kld = kld_loss(z_mu, z_logvar)

        loss = recon_loss + cost_loss + constraint_loss + integrality_loss + beta * kld
        loss.backward()
        optimizer.step()

        bs = data_obj_batch.num_graphs
        total_loss += loss.item() * bs
        total_recon_loss += recon_loss.item() * bs
        total_cost_loss += cost_loss.item() * bs
        total_constraint_loss += constraint_loss.item() * bs
        total_integrality_loss += integrality_loss.item() * bs
        total_kld_loss += kld.item() * bs

    n = len(train_loader.dataset)
    return (total_loss/n, total_recon_loss/n, total_cost_loss/n, total_constraint_loss/n, total_integrality_loss/n, total_kld_loss/n)


def test(loader, model, criterion_mse, criterion_bce, device='cuda', beta=1e-3):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_cost_loss = 0
    total_constraint_loss = 0
    total_integrality_loss = 0
    total_kld_loss = 0

    with torch.no_grad():
        for data_obj_batch, data_feas_batch in loader:
            data_obj_batch = data_obj_batch.to(device)
            data_feas_batch = data_feas_batch.to(device)

            x_hat, predicted_cost, predicted_constraints, predicted_integrality, z_mu, z_logvar = model(data_obj_batch, data_feas_batch)

            y_x = data_obj_batch.y_x
            y_cost = data_obj_batch.y_cost.squeeze()
            y_constraints = data_feas_batch.y_constraints
            y_integrality = data_obj_batch.y_integrality.to(device)

            recon_loss = criterion_mse(x_hat, y_x)
            cost_loss = criterion_mse(predicted_cost, y_cost)
            constraint_loss = criterion_mse(predicted_constraints, y_constraints)
            integrality_loss = criterion_bce(predicted_integrality, y_integrality)

            kld = kld_loss(z_mu, z_logvar)

            loss = recon_loss + cost_loss + constraint_loss + integrality_loss + beta * kld

            bs = data_obj_batch.num_graphs
            total_loss += loss.item() * bs
            total_recon_loss += recon_loss.item() * bs
            total_cost_loss += cost_loss.item() * bs
            total_constraint_loss += constraint_loss.item() * bs
            total_integrality_loss += integrality_loss.item() * bs
            total_kld_loss += kld.item() * bs

    n = len(loader.dataset)
    return (total_loss/n, total_recon_loss/n, total_cost_loss/n, total_constraint_loss/n, total_integrality_loss/n, total_kld_loss/n)
