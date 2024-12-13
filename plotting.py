import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_losses(
    num_epochs,
    train_total_losses, train_recon_losses, train_cost_losses, train_constraint_losses, train_integrality_losses, train_kld_losses,
    test_total_losses, test_recon_losses, test_cost_losses, test_constraint_losses, test_integrality_losses, test_kld_losses
):
    """
    Plots training and testing losses over epochs side by side in a single row.
    Saves the plot to a high-quality PNG file.

    Parameters:
    -----------
    num_epochs : int
        The total number of epochs trained.
    train_*_losses : list[float]
        Training losses per epoch for different metrics.
    test_*_losses : list[float]
        Testing losses per epoch for different metrics (including initial test evaluation at epoch 0).

    The losses arrays must include the initial test evaluation at epoch 0 for the test_*_losses arrays.
    The train_*_losses arrays start at epoch 1.
    """

    # Create epoch ranges
    epochs = range(num_epochs + 1)  # Includes the initial test eval (at epoch 0)
    train_epochs = range(1, num_epochs + 1)  # Training losses start from epoch 1

    # Use a nice style and context
    sns.set_style("whitegrid")
    sns.set_context("talk")  # Larger font size
    palette = sns.color_palette("colorblind")

    fig, axs = plt.subplots(1, 6, figsize=(30, 5))

    plot_kwargs_train = {"color": palette[0], "marker": "o", "markersize": 5, "linewidth": 2}
    plot_kwargs_test = {"color": palette[1], "marker": "s", "markersize": 5, "linewidth": 2}

    # Total Loss
    axs[0].plot(train_epochs, train_total_losses, **plot_kwargs_train, label='Train Total Loss')
    axs[0].plot(epochs, test_total_losses, **plot_kwargs_test, label='Test Total Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Total Loss')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid(True, which='both', linestyle='--')

    # Reconstruction Loss
    axs[1].plot(train_epochs, train_recon_losses, **plot_kwargs_train, label='Train Recon Loss')
    axs[1].plot(epochs, test_recon_losses, **plot_kwargs_test, label='Test Recon Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Reconstruction Loss')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].grid(True, which='both', linestyle='--')

    # Cost Loss
    axs[2].plot(train_epochs, train_cost_losses, **plot_kwargs_train, label='Train Cost Loss')
    axs[2].plot(epochs, test_cost_losses, **plot_kwargs_test, label='Test Cost Loss')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].set_title('Cost Loss')
    axs[2].set_yscale('log')
    axs[2].legend()
    axs[2].grid(True, which='both', linestyle='--')

    # Constraint Loss
    axs[3].plot(train_epochs, train_constraint_losses, **plot_kwargs_train, label='Train Constraint Loss')
    axs[3].plot(epochs, test_constraint_losses, **plot_kwargs_test, label='Test Constraint Loss')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Loss')
    axs[3].set_title('Constraint Loss')
    axs[3].set_yscale('log')
    axs[3].legend()
    axs[3].grid(True, which='both', linestyle='--')

    # Integrality Loss
    axs[4].plot(train_epochs, train_integrality_losses, **plot_kwargs_train, label='Train Integrality Loss')
    axs[4].plot(epochs, test_integrality_losses, **plot_kwargs_test, label='Test Integrality Loss')
    axs[4].set_xlabel('Epoch')
    axs[4].set_ylabel('Loss')
    axs[4].set_title('Integrality Loss')
    axs[4].set_yscale('log')
    axs[4].legend()
    axs[4].grid(True, which='both', linestyle='--')

    # KLD Loss
    axs[5].plot(train_epochs, train_kld_losses, **plot_kwargs_train, label='Train KLD Loss')
    axs[5].plot(epochs, test_kld_losses, **plot_kwargs_test, label='Test KLD Loss')
    axs[5].set_xlabel('Epoch')
    axs[5].set_ylabel('Loss')
    axs[5].set_title('KLD Loss')
    axs[5].set_yscale('log')
    axs[5].legend()
    axs[5].grid(True, which='both', linestyle='--')

    # Remove top and right spines
    for ax in axs:
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    plt.tight_layout()

    # Save the figure at high resolution (e.g., 300 dpi)
    plt.savefig("losses_plot.png", dpi=300, bbox_inches='tight')

    plt.show()

def plot_predictions_vs_actuals(model, test_loader, device='cuda', num_samples_to_show=1000):
    model.eval()
    all_actual_costs = []
    all_predicted_costs = []
    all_actual_constraints = []
    all_predicted_constraints = []

    samples_shown = 0
    with torch.no_grad():
        for data_obj_batch, data_feas_batch in test_loader:
            data_obj_batch = data_obj_batch.to(device)
            data_feas_batch = data_feas_batch.to(device)

            # Forward pass
            x_hat, predicted_cost, predicted_constraints, predicted_integrality = model(data_obj_batch, data_feas_batch)

            actual_costs_batch = data_obj_batch.y_cost.squeeze().cpu().numpy()
            predicted_costs_batch = predicted_cost.cpu().numpy()

            actual_constraints_batch = data_feas_batch.y_constraints.cpu().numpy()  # Actual constraint violations
            predicted_constraints_batch = predicted_constraints.cpu().numpy()        # Predicted constraint violations

            # Store the results
            for i in range(data_obj_batch.num_graphs):
                all_actual_costs.append(actual_costs_batch[i])
                all_predicted_costs.append(predicted_costs_batch[i])
                all_actual_constraints.append(actual_constraints_batch[i])
                all_predicted_constraints.append(predicted_constraints_batch[i])

                samples_shown += 1
                if samples_shown >= num_samples_to_show:
                    break
            if samples_shown >= num_samples_to_show:
                break

    # Convert to numpy arrays
    all_actual_costs = np.array(all_actual_costs)
    all_predicted_costs = np.array(all_predicted_costs)
    all_actual_constraints = np.array(all_actual_constraints)
    all_predicted_constraints = np.array(all_predicted_constraints)

    # Create side-by-side subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot for costs
    axs[0].scatter(all_actual_costs, all_predicted_costs, alpha=0.7, edgecolors='k')
    axs[0].plot([all_actual_costs.min(), all_actual_costs.max()],
                [all_actual_costs.min(), all_actual_costs.max()],
                color='red', linestyle='--', linewidth=2)  # line y=x
    axs[0].set_xlabel("Actual Cost")
    axs[0].set_ylabel("Predicted Cost")
    axs[0].set_title("Predicted vs Actual Costs")
    axs[0].grid(True)

    # Scatter plot for constraints
    axs[1].scatter(all_actual_constraints, all_predicted_constraints, alpha=0.7, edgecolors='k')
    axs[1].plot([all_actual_constraints.min(), all_actual_constraints.max()],
                [all_actual_constraints.min(), all_actual_constraints.max()],
                color='red', linestyle='--', linewidth=2)  # line y=x
    axs[1].set_xlabel("Actual Constraint Violation")
    axs[1].set_ylabel("Predicted Constraint Violation")
    axs[1].set_title("Predicted vs Actual Constraint Violations")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()