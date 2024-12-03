import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, continuous_size, binary_size):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.fc_continuous = nn.Linear(hidden_size, continuous_size)
        self.fc_binary = nn.Linear(hidden_size, binary_size)
        # No activation here; we'll add it in the forward method

    def forward(self, z):
        h = self.fc(z)
        x_continuous = self.fc_continuous(h)
        x_binary_logits = self.fc_binary(h)
        # Apply Sigmoid activation to binary outputs
        x_binary = torch.sigmoid(x_binary_logits)
        # Return concatenated outputs
        x_recon = torch.cat([x_continuous, x_binary], dim=1)
        return x_recon

# Define Predictors for Constraint Violations
class InequalityViolationPredictor(nn.Module):
    def __init__(self, latent_size, num_inequality_constraints):
        super(InequalityViolationPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, num_inequality_constraints)
        )

    def forward(self, z):
        return self.fc(z)

class EqualityViolationPredictor(nn.Module):
    def __init__(self, latent_size, num_equality_constraints):
        super(EqualityViolationPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, num_equality_constraints)
        )

    def forward(self, z):
        return self.fc(z)

# In networks.py
class CostPredictor(nn.Module):
    def __init__(self, latent_size):
        super(CostPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, z):
        return self.model(z)

