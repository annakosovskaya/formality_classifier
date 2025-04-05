import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
import os

def plot_and_print_results(
    results_df: pd.DataFrame, 
    save_path: Optional[str] = None,
):
    """
    Plot and print evaluation results.
    
    Args:
        results_df: DataFrame containing evaluation results
        save_path: Optional path to save the plot image. If None, plot won't be saved.
    """
    # Sort by number of shots to ensure correct plotting
    results_df = results_df.sort_values("num_shots")

    # Create 3 side-by-side plots: Accuracy, Formal F1, Informal F1
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    # Accuracy Plot
    axes[0].plot(results_df["num_shots"], results_df["accuracy"], marker="o")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Number of Shots")
    axes[0].set_ylabel("Score")
    axes[0].grid(True)
    axes[0].set_ylim(0, 1)

    # Formal F1 Plot
    axes[1].plot(results_df["num_shots"], results_df["formal_f1"], marker="o")
    axes[1].set_title("Formal F1 Score")
    axes[1].set_xlabel("Number of Shots")
    axes[1].grid(True)
    axes[1].set_ylim(0, 1)

    # Informal F1 Plot
    axes[2].plot(results_df["num_shots"], results_df["informal_f1"], marker="o")
    axes[2].set_title("Informal F1 Score")
    axes[2].set_xlabel("Number of Shots")
    axes[2].grid(True)
    axes[2].set_ylim(0, 1)

    # Adjust layout and save
    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")

    plt.show()

    # Print Final Results
    print("\nFinal Results (Batched Evaluation):")
    for _, row in results_df.iterrows():
        print(f"{row['num_shots']}-shot → "
              f"accuracy: {row['accuracy']:.4f} | "
              f"formal F1: {row['formal_f1']:.4f} | "
              f"informal F1: {row['informal_f1']:.4f}") 