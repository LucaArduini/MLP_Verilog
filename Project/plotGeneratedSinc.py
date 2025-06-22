import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting

def plot_mlp_predictions(filename="inference_grid_results.txt"):
    """
    Loads data from the inference results file and plots the 3D surface.
    Also plots the true sinc(x1)*sinc(x2) surface.
    """
    try:
        data = np.loadtxt(filename)
    except IOError:
        print(f"Error: Could not load data from {filename}. Make sure the C++ program ran successfully.")
        return
    except ValueError:
        print(f"Error: Could not parse data from {filename}. Check the file format.")
        return


    x1_flat = data[:, 0]
    x2_flat = data[:, 1]
    y_pred_flat = data[:, 2]

    # Infer grid dimensions (assuming a square grid was generated)
    # If not a square grid, this needs to be known beforehand.
    total_points = len(x1_flat)
    grid_points_per_dim = int(np.sqrt(total_points))

    if grid_points_per_dim * grid_points_per_dim != total_points:
        print("Warning: The number of data points does not form a perfect square grid.")
        # Attempt to reshape anyway, might fail or produce incorrect plot
        # For non-square grids, you'd need to know the original dimensions.
        # For now, we proceed assuming it's meant to be square or C-order flattened.
        # A more robust solution would be to save grid dimensions to the file or pass them.
        
    # Reshape for plotting. Assumes data is stored row-by-row (x1 varies fastest).
    X1 = x1_flat.reshape((grid_points_per_dim, grid_points_per_dim))
    X2 = x2_flat.reshape((grid_points_per_dim, grid_points_per_dim))
    Y_pred = y_pred_flat.reshape((grid_points_per_dim, grid_points_per_dim))

    # --- Plot MLP Predicted Surface ---
    fig_pred = plt.figure(figsize=(12, 9))
    ax_pred = fig_pred.add_subplot(111, projection='3d')
    surf_pred = ax_pred.plot_surface(X1, X2, Y_pred, cmap='viridis', edgecolor='none')
    fig_pred.colorbar(surf_pred, shrink=0.5, aspect=10)

    ax_pred.set_xlabel('x1')
    ax_pred.set_ylabel('x2')
    ax_pred.set_zlabel('Predicted MLP Output')
    ax_pred.set_title('MLP Predicted Surface')
    ax_pred.view_init(elev=30, azim=-60) # Adjust view angle for better visualization

    # --- Plot True Sinc Function Surface ---
    def true_sinc2d_func(x1_val, x2_val):
        # Definition: sinc(x) = sin(x)/x for x != 0, and sinc(0) = 1
        sinc_x1 = np.where(x1_val == 0, 1.0, np.sin(x1_val) / x1_val)
        sinc_x2 = np.where(x2_val == 0, 1.0, np.sin(x2_val) / x2_val)
        return 10.0 * sinc_x1 * sinc_x2

    Y_true = true_sinc2d_func(X1, X2)

    fig_true = plt.figure(figsize=(12, 9))
    ax_true = fig_true.add_subplot(111, projection='3d')
    surf_true = ax_true.plot_surface(X1, X2, Y_true, cmap='coolwarm', edgecolor='none')
    fig_true.colorbar(surf_true, shrink=0.5, aspect=10)
    
    ax_true.set_xlabel('x1')
    ax_true.set_ylabel('x2')
    ax_true.set_zlabel('True Output (10*sinc(x1)*sinc(x2))')
    ax_true.set_title('True Sinc Function Surface')
    ax_true.view_init(elev=30, azim=-60) # Match view angle

    plt.show()

if __name__ == '__main__':
    plot_mlp_predictions()