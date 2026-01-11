"""
Evaluation and visualization utilities.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path (str): Optional path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {full_path}")
    
    plt.close()


def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot feature importance.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        save_path (str): Optional path to save the plot
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if save_path:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {full_path}")
    
    plt.close()


def print_evaluation_summary(metrics):
    """
    Print evaluation summary.
    
    Args:
        metrics (dict): Dictionary with evaluation metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION SUMMARY")
    print("="*50)
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Overfitting Gap: {metrics['overfitting_gap']:.4f}")
    
    if metrics['overfitting_gap'] > 0.1:
        print("⚠️  WARNING: Large gap detected - possible overfitting!")
    else:
        print("✓ Model generalization looks good")
    
    print("\nClassification Report:")
    print(classification_report_to_string(metrics['classification_report']))
    print("="*50 + "\n")


def classification_report_to_string(report_dict):
    """
    Convert classification report dict to formatted string.
    """
    lines = []
    lines.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    lines.append("-" * 65)
    
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            lines.append(
                f"{class_name:<15} {metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f} "
                f"{int(metrics['support']):<10}"
            )
    
    # Add accuracy
    if 'accuracy' in report_dict:
        lines.append("-" * 65)
        lines.append(f"{'accuracy':<15} {'':<12} {'':<12} {report_dict['accuracy']:<12.4f}")
    
    return "\n".join(lines)

