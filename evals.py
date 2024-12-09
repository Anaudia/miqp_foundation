from scipy.stats import pearsonr
import pandas as pd
from IPython.display import display
import numpy as np
import torch

def show_constraint_predictions(model, data_list, n, m, p, target_mean, target_std, loader_name="Training"):
    model.eval()
    print(f"\n{loader_name} Set Predictions:")
    with torch.no_grad():
        # Only display results for the first two samples
        for idx, data in enumerate(data_list[:2]):
            out = model(data)
            # Get indices of constraint nodes (node indices start from 0 in each graph)
            inequality_constraint_indices = torch.arange(n, n + m)
            equality_constraint_indices = torch.arange(n + m, n + m + p)
            # Predicted violations
            pred_inequality = out[inequality_constraint_indices].view(-1)
            pred_equality = out[equality_constraint_indices].view(-1)
            # Actual violations
            actual_inequality = data.y[:m]
            actual_equality = data.y[m:]
            # De-normalize predictions and actual values
            pred_inequality = pred_inequality * target_std + target_mean
            pred_equality = pred_equality * target_std + target_mean
            actual_inequality = actual_inequality * target_std + target_mean
            actual_equality = actual_equality * target_std + target_mean
            # Prepare data for DataFrame
            print(f"\nSample {idx+1}:")
            # Inequality Constraints DataFrame
            inequality_data = {
                'Constraint': [f'Inequality {j+1}' for j in range(m)],
                'Actual Violation': actual_inequality.cpu().numpy(),
                'Predicted Violation': pred_inequality.cpu().numpy()
            }
            inequality_df = pd.DataFrame(inequality_data)
            print("Inequality Constraints:")
            display(inequality_df)
            # Equality Constraints DataFrame
            if p > 0:
                equality_data = {
                    'Constraint': [f'Equality {j+1}' for j in range(p)],
                    'Actual Violation': actual_equality.cpu().numpy(),
                    'Predicted Violation': pred_equality.cpu().numpy()
                }
                equality_df = pd.DataFrame(equality_data)
                print("Equality Constraints:")
                display(equality_df)


# Function to calculate correlation between predictions and actual values
def calculate_correlation(loader, model, target_std, target_mean):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            # De-normalize predictions and targets
            pred_cost = out.view(-1) * target_std + target_mean
            actual_cost = data.y.view(-1) * target_std + target_mean
            predictions.append(pred_cost.cpu().numpy())
            actuals.append(actual_cost.cpu().numpy())

    # Flatten the lists and compute Pearson correlation
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    correlation, _ = pearsonr(predictions, actuals)
    return correlation