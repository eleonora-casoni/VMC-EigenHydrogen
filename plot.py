import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_histogram(position_vec_fin, save_path=None):
    """
    Generate a histogram of final positions.

    Args:
        position_vec_fin (np.ndarray): Array of final walker positions.
        save_path (str): Path to save the plot. 

    Returns:
        None
    """
    plt.figure()
    plt.hist(position_vec_fin, bins=200, density=True, alpha=0.6, color="g")
    plt.xlabel("Position")
    plt.ylabel("Density")
    plt.title("Histogram of Final Positions")
    plt.grid(True)
   
    plt.savefig(save_path, dpi=300)
    print(f"Saved final positions to {save_path}")
    


def plot_alpha_evolution(alpha_buffer, save_path=None):
    """
    Generate and optionally save a plot of the evolution of alpha.

    Args:
        alpha_buffer (np.ndarray): Array of alpha values over time.
        save_path (str): Path to save the plot. 

    Returns:
        None
    """
    plt.figure()
    plt.plot(range(len(alpha_buffer)), alpha_buffer, "r", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Alpha")
    plt.title("Evolution of Alpha")
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300)
    print(f"Saved alpha evolution plot to {save_path}")
    


def plot_energy_evolution(E_buffer, save_path=None):
    """
    Generate and optionally save a plot of the evolution of energy.

    Args:
        E_buffer (np.ndarray): Array of energy values over time.
        save_path (str): Path to save the plot. 

    Returns:
        None
    """
    plt.figure()
    plt.plot(range(len(E_buffer)), E_buffer, "b", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Evolution of Energy")
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300)
    print(f"Saved energy evolution plot to {save_path}")
   


def save_results_to_csv(position_vec_fin, alpha_buffer, E_buffer, output_dir="results"):
    """
    Save simulation results to CSV files.

    Args:
        position_vec_fin (np.ndarray): Final walker positions.
        alpha_buffer (np.ndarray): Evolution of alpha values.
        E_buffer (np.ndarray): Evolution of energy values.
        output_dir (str): Directory to save the CSV files.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame({"position": position_vec_fin}).to_csv(
        f"{output_dir}/final_positions.csv", index=False
    )
    pd.DataFrame({"alpha": alpha_buffer}).to_csv(
        f"{output_dir}/alpha_evolution.csv", index=False
    )
    pd.DataFrame({"energy": E_buffer}).to_csv(
        f"{output_dir}/energy_evolution.csv", index=False
    )

    print(f"Values saved to {output_dir}")
