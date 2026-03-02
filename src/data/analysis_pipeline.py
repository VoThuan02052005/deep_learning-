import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.logger import Logger

logger = Logger("analysis_pipeline")

def load_data(file_path):
    """Load data from CSV file."""
    logger.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def plot_skew_check_and_save(df, col, save_dir="reports/figures", bins=50):
    """
    Check skewness and save Springer-style figures.
    Ported from analysis.ipynb.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    data_val = df[col].dropna().values
    log_data = np.log1p(data_val)

    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Original distribution (clipped at 99th percentile)
    upper = np.percentile(data_val, 99)
    axes[0].hist(data_val, bins=bins, color="lightgray", edgecolor="black", linewidth=0.8)
    axes[0].set_xlim(0, upper)
    axes[0].set_title(f"Original {col} Distribution (99% clipped)")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frequency")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Log-transformed distribution
    axes[1].hist(log_data, bins=bins, color="lightgray", edgecolor="black", linewidth=0.8)
    axes[1].set_title(f"Log-transformed {col} Distribution")
    axes[1].set_xlabel(f"log(1 + {col})")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    file_path = os.path.join(save_dir, f"{col}_log_transform_springer.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure to: {file_path}")

def perform_analysis(data_path, output_dir="reports/figures"):
    """Main analysis pipeline execution."""
    data = load_data(data_path)
    logger.info(f"Data shape: {data.shape}")
    
    # Perform skew check for numerical columns as seen in analysis.ipynb
    if "area" in data.columns:
        plot_skew_check_and_save(data, "area", save_dir=output_dir)
    if "price" in data.columns:
        plot_skew_check_and_save(data, "price", save_dir=output_dir)
        
    logger.success("Analysis pipeline completed successfully")

if __name__ == "__main__":
    # Default path for testing
    DATA_PATH = "data/staging/data_sau_clean.csv"
    perform_analysis(DATA_PATH)
