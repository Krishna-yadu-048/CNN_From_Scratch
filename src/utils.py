"""
Utility functions for training, evaluation, and visualization.
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


# ========== Training Helpers ==========

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch. Returns avg loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Run validation. Returns avg loss, accuracy, all predictions and labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            running_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ========== Visualization Functions ==========

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Plot training and validation loss/accuracy curves side by side.
    Uses Plotly for interactive charts.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss Over Epochs", "Accuracy Over Epochs"),
        horizontal_spacing=0.12
    )
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Loss curves
    fig.add_trace(
        go.Scatter(x=epochs, y=train_losses, mode='lines+markers',
                   name='Train Loss', line=dict(color='#FF6B6B', width=2),
                   marker=dict(size=5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_losses, mode='lines+markers',
                   name='Val Loss', line=dict(color='#4ECDC4', width=2),
                   marker=dict(size=5)),
        row=1, col=1
    )
    
    # Accuracy curves
    fig.add_trace(
        go.Scatter(x=epochs, y=train_accs, mode='lines+markers',
                   name='Train Acc', line=dict(color='#FF6B6B', width=2),
                   marker=dict(size=5), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_accs, mode='lines+markers',
                   name='Val Acc', line=dict(color='#4ECDC4', width=2),
                   marker=dict(size=5), showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=420,
        width=900,
        title_text="Training Progress",
        title_x=0.5,
        font=dict(family="Segoe UI, sans-serif", size=13),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)")
    )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=["Cat", "Dog"]):
    """
    Plot a confusion matrix using seaborn.
    Returns the matplotlib figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 18, "weight": "bold"},
        linewidths=2, linecolor='white',
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=13, fontweight='bold')
    ax.set_ylabel("Actual", fontsize=13, fontweight='bold')
    ax.set_title("Confusion Matrix", fontsize=15, fontweight='bold', pad=12)
    plt.tight_layout()
    
    return fig


def plot_roc_curve(y_true, y_probs):
    """Plot ROC curve with AUC score using Plotly."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#FF6B6B', width=3)
    ))
    
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title=dict(text="ROC Curve", x=0.5),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450, width=550,
        font=dict(family="Segoe UI, sans-serif", size=13),
        legend=dict(x=0.4, y=0.1, bgcolor="rgba(0,0,0,0.3)")
    )
    
    return fig


def plot_prediction_distribution(y_probs, y_true):
    """
    Show how confident the model is — histogram of predicted probabilities
    split by actual class.
    """
    fig = go.Figure()
    
    cat_probs = y_probs[y_true == 0]
    dog_probs = y_probs[y_true == 1]
    
    fig.add_trace(go.Histogram(
        x=cat_probs, name="Actual: Cat",
        marker_color='#4ECDC4', opacity=0.7,
        nbinsx=30
    ))
    fig.add_trace(go.Histogram(
        x=dog_probs, name="Actual: Dog",
        marker_color='#FF6B6B', opacity=0.7,
        nbinsx=30
    ))
    
    fig.update_layout(
        template="plotly_dark",
        barmode='overlay',
        title=dict(text="Prediction Confidence Distribution", x=0.5),
        xaxis_title="Predicted Probability (Dog)",
        yaxis_title="Count",
        height=420, width=700,
        font=dict(family="Segoe UI, sans-serif", size=13),
        legend=dict(x=0.75, y=0.95, bgcolor="rgba(0,0,0,0.3)")
    )
    
    return fig


def get_classification_metrics(y_true, y_pred):
    """Return classification report as a dict for display."""
    report = classification_report(y_true, y_pred,
                                   target_names=["Cat", "Dog"],
                                   output_dict=True)
    return report
