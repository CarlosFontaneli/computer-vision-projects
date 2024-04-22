import matplotlib.pyplot as plt
import torch


def create_image(func, inputs, eps=0.1, grid_resolution=300):
    """Create an image containing the output of a function for each point
    of a grid. Used for visualizing the decision surface of a model."""

    x0, x1 = inputs.T

    x0_min, x0_max = x0.min() - eps, x0.max() + eps
    x1_min, x1_max = x1.min() - eps, x1.max() + eps

    xx0, xx1 = torch.meshgrid(
        torch.linspace(x0_min, x0_max, grid_resolution),
        torch.linspace(x1_min, x1_max, grid_resolution),
        indexing="xy",
    )

    data_grid = torch.stack((xx0.reshape(-1), xx1.reshape(-1)), dim=1)

    response = func(data_grid)
    response = response.reshape(xx0.shape)

    return response, xx0, xx1


def plot_regions(model, inputs, targets, grid_resolution=300, eps=0.5):
    """Plot the output of a model for a dense grid of points bounded by the data."""

    def get_probs(inputs):
        with torch.no_grad():
            scores = model(inputs)
        return 1 / (
            1 + torch.exp(-scores)
        )  # Sigmoid to convert logits to probabilities

    response, xx0, xx1 = create_image(
        get_probs, inputs, eps=eps, grid_resolution=grid_resolution
    )

    plt.rcParams.update({"font.size": 22})

    fig, ax = plt.subplots(figsize=(15, 10))

    # Improved color mapping for better visual distinction
    cmap = plt.get_cmap("viridis_r")
    co = ax.pcolormesh(
        xx0, xx1, response, cmap=cmap, shading="auto"
    )  # Use 'auto' shading for a better color gradient

    # Increase scatter size and add edge color for better visibility
    scatter0 = ax.scatter(*inputs[targets == 0].T, s=50, label="Class 0")
    scatter1 = ax.scatter(*inputs[targets == 1].T, s=50, label="Class 1")

    # Adding a legend to help identify classes
    ax.legend(handles=[scatter0, scatter1], title="Classes", loc="upper right")

    # Adding a color bar for the probability scale
    cbar = fig.colorbar(co, ax=ax, label="P(Class=1 | Features)")
    cbar.set_label("Probability of Class 1", rotation=270, labelpad=20)

    # Set labels and title for clarity
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Surface with Class Distributions")

    plt.show()
