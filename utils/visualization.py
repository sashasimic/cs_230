"""Visualization utilities."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List
import os

plt.style.use('seaborn-v0_8-darkgrid')


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        figsize: Figure size
    """
    metrics = [k for k in history.keys() if not k.startswith('val_')]
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], 'b-', label=f'Training {metric}', linewidth=2)
        
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(epochs, history[val_metric], 'r-', label=f'Validation {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} over Epochs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'Predictions vs Actual',
    figsize: tuple = (15, 10)
):
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Time series plot
    ax1 = axes[0]
    indices = np.arange(len(y_true))
    ax1.plot(indices, y_true, 'b-', label='Actual', linewidth=1.5, alpha=0.7)
    ax1.plot(indices, y_pred, 'r-', label='Predicted', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Sample', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(f'{title} - Time Series', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Values', fontsize=12)
    ax2.set_ylabel('Predicted Values', fontsize=12)
    ax2.set_title(f'{title} - Scatter Plot', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add metrics to plot
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    textstr = f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'Feature Importance',
    top_k: int = 20,
    figsize: tuple = (10, 8)
):
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        save_path: Path to save figure
        title: Plot title
        top_k: Number of top features to show
        figsize: Figure size
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_k]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importances, align='center', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
