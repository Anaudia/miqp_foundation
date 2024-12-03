import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, continuous_size, binary_size):
        super(Decoder, self).__init__()
        self.fc_common = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU()
        )
        self.fc_continuous = nn.Linear(hidden_size, continuous_size)
        self.fc_binary = nn.Linear(hidden_size, binary_size)

    def forward(self, z):
        h = self.fc_common(z)
        x_continuous = self.fc_continuous(h)  # Linear activation
        x_binary = torch.sigmoid(self.fc_binary(h))  # Sigmoid activation
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

class CostPredictor(nn.Module):
    def __init__(self, latent_size):
        super(CostPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, 1)
        )

    def forward(self, z):
        return self.fc(z)
