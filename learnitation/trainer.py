import copy
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10,
    learning_rate=0.001,
    save_last_snapshots=False,
):
    # Check for available device - now including MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # List to store model snapshots from last epoch
    model_snapshots = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Save model snapshot if in the last epoch
            if save_last_snapshots and epoch == num_epochs - 1:
                model_copy = copy.deepcopy(model)
                model_copy.eval()  # Set to eval mode for consistency
                model_snapshots.append(model_copy)

        train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss = test_loss / len(test_loader)
        test_accuracy = 100.0 * correct / total

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print("--------------------")

    results = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
    }

    if save_last_snapshots:
        results["model_snapshots"] = model_snapshots

    return results


def pure_test(model, test_loader):
    # Check for available device - now including MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.NLLLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data to device
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100.0 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


def get_flattened_parameters(model):
    """Convert all model parameters into a single flat vector"""
    return torch.cat([p.data.flatten() for p in model.parameters()])


def calculate_l2_distances(models):
    """
    Calculate pairwise L2 distances between a list of models

    Args:
        models: List of PyTorch models with the same architecture

    Returns:
        distances: Dictionary of pairwise distances with tuple keys (i,j)
        param_distances: Dictionary of per-parameter distances (optional)
    """
    n_models = len(models)
    distances = {}
    param_distances = {}

    # Get flattened parameters for each model
    flat_params = [get_flattened_parameters(model) for model in models]

    # Calculate pairwise distances
    for i, j in itertools.combinations(range(n_models), 2):
        diff = flat_params[i] - flat_params[j]
        l2_dist = torch.norm(diff).item()
        distances[(i, j)] = l2_dist

        # Optional: Calculate per-parameter distances
        param_dists = []
        params_i = list(models[i].parameters())
        params_j = list(models[j].parameters())

        for p1, p2 in zip(params_i, params_j):
            param_diff = p1.data - p2.data
            param_dist = torch.norm(param_diff).item()
            param_dists.append(param_dist)

        param_distances[(i, j)] = param_dists

    return distances, param_distances


def print_distance_matrix(models, distances):
    """Pretty print the distance matrix"""
    n_models = len(models)
    print("\nL2 Distance Matrix:")
    print(" " * 8, end="")
    for i in range(n_models):
        print(f"Model {i:2}", end="  ")
    print()

    for i in range(n_models):
        print(f"Model {i:2}", end="  ")
        for j in range(n_models):
            if i == j:
                print(f"{0:8.3f}", end="  ")
            elif (i, j) in distances:
                print(f"{distances[(i,j)]:8.3f}", end="  ")
            else:
                print(f"{distances[(j,i)]:8.3f}", end="  ")
        print()


def analyze_parameter_differences(models, param_distances):
    """Analyze which layers contribute most to the differences"""
    named_params = list(models[0].named_parameters())
    n_models = len(models)

    print("\nPer-layer contribution to differences:")
    for (name, _), layer_idx in zip(named_params, range(len(named_params))):
        print(f"\n{name}:")
        for i, j in itertools.combinations(range(n_models), 2):
            layer_dist = param_distances[(i, j)][layer_idx]
            print(f"  Model {i} vs Model {j}: {layer_dist:.3f}")


def get_weight_vector(model):
    """Convert model parameters to a single numpy vector"""
    return torch.cat([p.data.flatten() for p in model.parameters()]).cpu().numpy()


def perform_model_pca(models, n_components=3):
    """
    Perform PCA on model weights

    Args:
        models: List of PyTorch models
        n_components: Number of PCA components to compute
    """
    # Get weight vectors for each model
    weight_vectors = np.array([get_weight_vector(model) for model in models])

    # Perform PCA
    pca = PCA(n_components=min(n_components, len(models)))
    transformed = pca.fit_transform(weight_vectors)

    # Create visualizations
    fig = plt.figure(figsize=(15, 5))

    # 2D Plot
    ax1 = fig.add_subplot(121)
    ax1.scatter(transformed[:, 0], transformed[:, 1], c="blue", marker="o", s=100)
    for i, (x, y) in enumerate(zip(transformed[:, 0], transformed[:, 1])):
        ax1.annotate(f"Model {i}", (x, y), xytext=(5, 5), textcoords="offset points")
    ax1.set_xlabel("First Principal Component")
    ax1.set_ylabel("Second Principal Component")
    ax1.set_title("PCA of Model Weights (2D)")
    ax1.grid(True)

    # 3D Plot if we have enough components
    if transformed.shape[1] >= 3:
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(
            transformed[:, 0],
            transformed[:, 1],
            transformed[:, 2],
            c="blue",
            marker="o",
            s=100,
        )
        for i, (x, y, z) in enumerate(
            zip(transformed[:, 0], transformed[:, 1], transformed[:, 2])
        ):
            ax2.text(x, y, z, f"Model {i}")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_zlabel("PC3")
        ax2.set_title("PCA of Model Weights (3D)")

    plt.tight_layout()

    # Print explained variance ratios
    print("\nExplained variance ratios:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.4f}")

    return pca, transformed


# Alternative visualization focusing on pairwise relationships
def plot_pca_pairwise(models):
    """Create a pairwise plot of the first 3 PCA components"""
    weight_vectors = np.array([get_weight_vector(model) for model in models])
    pca = PCA(n_components=3)
    transformed = pca.fit_transform(weight_vectors)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    components = ["PC1", "PC2", "PC3"]

    for i in range(3):
        for j in range(3):
            if i != j:
                axes[i, j].scatter(
                    transformed[:, j], transformed[:, i], c="blue", s=100
                )
                for k, (x, y) in enumerate(zip(transformed[:, j], transformed[:, i])):
                    axes[i, j].annotate(
                        f"Model {k}", (x, y), xytext=(5, 5), textcoords="offset points"
                    )
                axes[i, j].set_xlabel(components[j])
                axes[i, j].set_ylabel(components[i])
                axes[i, j].grid(True)
            else:
                axes[i, j].hist(transformed[:, i], bins="auto")
                axes[i, j].set_xlabel(components[i])
                axes[i, j].set_ylabel("Frequency")
                axes[i, j].grid(True)

    plt.tight_layout()
    return pca, transformed


def analyze_weight_spectrum(models):
    """
    Analyze the spectrum of model weight differences

    Args:
        models: List of PyTorch models
    Returns:
        pca: Fitted PCA object
        spectrum_data: Dictionary containing spectrum analysis results
    """
    # Get weight vectors for each model
    weight_vectors = np.array([get_weight_vector(model) for model in models])

    # Perform full PCA
    pca = PCA()
    pca.fit(weight_vectors)

    # Calculate key metrics
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Find number of components for different variance thresholds
    variance_thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
    components_needed = [
        np.argmax(cumulative_variance >= threshold) + 1
        for threshold in variance_thresholds
    ]

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Spectrum plot (log scale)
    ax1.plot(
        range(1, len(explained_variance) + 1),
        explained_variance,
        "bo-",
        label="Individual",
    )
    ax1.plot(
        range(1, len(explained_variance) + 1),
        cumulative_variance,
        "ro-",
        label="Cumulative",
    )
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("PCA Spectrum (Log Scale)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale("log")

    # Spectrum plot (linear scale)
    ax2.plot(
        range(1, len(explained_variance) + 1),
        explained_variance,
        "bo-",
        label="Individual",
    )
    ax2.plot(
        range(1, len(explained_variance) + 1),
        cumulative_variance,
        "ro-",
        label="Cumulative",
    )
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_title("PCA Spectrum (Linear Scale)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print analysis
    print("\nSpectrum Analysis:")
    print("-----------------")
    print("Top 5 Components:")
    for i in range(min(5, len(explained_variance))):
        print(
            f"PC{i+1}: {explained_variance[i]:.4f} "
            f"(Cumulative: {cumulative_variance[i]:.4f})"
        )

    print("\nComponents needed for variance explained:")
    for threshold, n_components in zip(variance_thresholds, components_needed):
        print(f"{threshold*100:3.0f}%: {n_components} components")

    # Calculate condition number (ratio of largest to smallest singular value)
    condition_number = np.sqrt(pca.explained_variance_[0] / pca.explained_variance_[-1])
    print(f"\nCondition Number: {condition_number:.2f}")

    # Return analysis results
    spectrum_data = {
        "explained_variance": explained_variance,
        "cumulative_variance": cumulative_variance,
        "components_needed": dict(zip(variance_thresholds, components_needed)),
        "condition_number": condition_number,
    }

    return pca, spectrum_data
