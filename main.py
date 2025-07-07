import torch
import torchvision
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
from PIL import Image
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, hamming_loss, precision_score, recall_score, 
    f1_score, classification_report, multilabel_confusion_matrix,
    jaccard_score, coverage_error, label_ranking_loss
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
root = '/media/mlr-lab/325C37DE7879ABF2/Lowlight_4_7_25/Lowlightdataset'
json_path = '/media/mlr-lab/325C37DE7879ABF2/Lowlight_4_7_25/Lowlightdataset/final_labels.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
EPOCHS = 100
model_output_dir = '/media/mlr-lab/325C37DE7879ABF2/Lowlight_4_7_25/Lowlightdataset/model_checkpoints'

# Create output directory if it doesn't exist
os.makedirs(model_output_dir, exist_ok=True)

class CustomDataset(Dataset):
    def __init__(self, root, json_path):
        super().__init__()
        with open(json_path) as f:
            self.data = json.load(f)
        self.root = root  # Fixed typo: was 'roota'
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.094, 0.091, 0.085], std=[0.091, 0.088, 0.087])
        ])

    def __len__(self):
        return len(list(self.data['pairs'].keys()))

    def __getitem__(self, idx):
        name = list(self.data['pairs'].keys())[idx]
        # Fixed path construction
        image_path = os.path.join(self.root, name.lstrip('/'))
        image = Image.open(image_path).convert('RGB')  # Ensure RGB format
        image = self.transform(image)
        label = torch.zeros(100)
        for i in self.data['pairs'][name]:
            label[i] = 1.0
        return image, label

def classifier(num_classes, device=DEVICE):
    predictor = nn.Linear(2048, num_classes).to(device)
    nn.init.xavier_uniform_(predictor.weight)
    nn.init.zeros_(predictor.bias)
    return predictor

def calculate_metrics(y_true, y_pred, y_scores=None):
    """Calculate comprehensive multi-label classification metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['hamming_accuracy'] = 1 - hamming_loss(y_true, y_pred)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    
    # Precision, Recall, F1 (micro and macro averages)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Jaccard similarity (IoU for multi-label)
    metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Ranking-based metrics (if scores are provided)
    if y_scores is not None:
        metrics['coverage_error'] = coverage_error(y_true, y_scores)
        metrics['label_ranking_loss'] = label_ranking_loss(y_true, y_scores)
    
    return metrics

def plot_confusion_matrices(y_true, y_pred, num_classes, save_path=None, top_k=10):
    """Plot confusion matrices for top-k most frequent labels"""
    # Calculate label frequencies
    label_counts = np.sum(y_true, axis=0)
    top_labels = np.argsort(label_counts)[-top_k:][::-1]
    
    # Create multilabel confusion matrix
    cm_multilabel = multilabel_confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrices for top-k labels
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for i, label_idx in enumerate(top_labels):
        cm = cm_multilabel[label_idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Label {label_idx}\n(Count: {int(label_counts[label_idx])})')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_label_distribution(y_true, y_pred, save_path=None):
    """Plot label distribution comparison"""
    true_counts = np.sum(y_true, axis=0)
    pred_counts = np.sum(y_pred, axis=0)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(true_counts, bins=20, alpha=0.7, label='True', color='blue')
    plt.hist(pred_counts, bins=20, alpha=0.7, label='Predicted', color='red')
    plt.xlabel('Label Count')
    plt.ylabel('Frequency')
    plt.title('Label Distribution Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(true_counts, pred_counts, alpha=0.6)
    plt.plot([0, max(true_counts)], [0, max(true_counts)], 'r--', alpha=0.8)
    plt.xlabel('True Label Counts')
    plt.ylabel('Predicted Label Counts')
    plt.title('True vs Predicted Label Counts')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, predictor, dataloader, device=DEVICE):
    """Comprehensive model evaluation"""
    model.eval()
    predictor.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            logits = predictor(features)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get predictions and scores
            scores = torch.sigmoid(logits).cpu().numpy()
            preds = (logits >= 0.5).float().cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels_np)
            all_scores.append(scores)
    
    # Concatenate all results
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_scores = np.vstack(all_scores)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_scores)
    metrics['avg_loss'] = total_loss / len(dataloader)
    
    return metrics, all_labels, all_preds, all_scores

def train(train_dataloader, test_dataloader, model, predictor, epochs, device=DEVICE):
    
    optimizer = AdamW([
        {'params': predictor.parameters(), 'lr': 1e-5},
        {'params': model.parameters(), 'lr': 1e-6}
    ])
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Training history
    train_losses = []
    val_losses = []
    val_metrics_history = []
    
    best_val_loss = float('inf')
    best_metrics = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        predictor.train()
        total_loss = 0
        
        loop = tqdm(train_dataloader, desc=f'Epoch: {epoch+1}/{epochs}')
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            features = model(images)
            pred = predictor(features)
            loss = criterion(pred, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_metrics, val_labels, val_preds, val_scores = evaluate_model(
            model, predictor, test_dataloader, device
        )
        
        val_losses.append(val_metrics['avg_loss'])
        val_metrics_history.append(val_metrics)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_metrics['avg_loss']:.4f}")
        print(f"Subset Accuracy: {val_metrics['subset_accuracy']:.4f}")
        print(f"Hamming Accuracy: {val_metrics['hamming_accuracy']:.4f}")
        print(f"F1 (Micro): {val_metrics['f1_micro']:.4f}")
        print(f"F1 (Macro): {val_metrics['f1_macro']:.4f}")
        print(f"Jaccard (Micro): {val_metrics['jaccard_micro']:.4f}")
        
        # Save best model
        if val_metrics['avg_loss'] < best_val_loss:
            best_val_loss = val_metrics['avg_loss']
            best_metrics = val_metrics
            print(f'Saving best model at Epoch {epoch+1}')
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['avg_loss'],
                'val_metrics': val_metrics,
                'train_loss': avg_train_loss
            }
            torch.save(checkpoint, os.path.join(model_output_dir, f'best_model_epoch_{epoch+1}.pt'))
            
            # Save detailed evaluation for best model
            if epoch == epochs - 1 or epoch % 10 == 0:  # Save detailed metrics periodically
                # Plot confusion matrices
                plot_confusion_matrices(
                    val_labels, val_preds, 100, 
                    save_path=os.path.join(model_output_dir, f'confusion_matrices_epoch_{epoch+1}.png')
                )
                
                # Plot label distribution
                plot_label_distribution(
                    val_labels, val_preds,
                    save_path=os.path.join(model_output_dir, f'label_distribution_epoch_{epoch+1}.png')
                )
                
                # Save detailed classification report
                with open(os.path.join(model_output_dir, f'classification_report_epoch_{epoch+1}.txt'), 'w') as f:
                    f.write("Multi-label Classification Report\n")
                    f.write("="*50 + "\n\n")
                    
                    for metric, value in val_metrics.items():
                        f.write(f"{metric}: {value:.4f}\n")
                    
                    f.write("\n" + "="*50 + "\n")
                    f.write("Per-label Classification Report:\n")
                    f.write(classification_report(val_labels, val_preds, zero_division=0))
        
        print("-" * 50)
    
    return best_metrics, train_losses, val_losses, val_metrics_history

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load dataset
    dataset = CustomDataset(root, json_path)
    print(f"Total dataset size: {len(dataset)}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Train size: {train_size}, Test size: {test_size}")
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=4)
    
    # Initialize model
    resnet = models.resnet50(weights='DEFAULT')
    resnet.fc = nn.Identity()
    resnet = resnet.to(DEVICE)
    
    predictor = classifier(num_classes=100)
    
    print(f"Model parameters: {sum(p.numel() for p in resnet.parameters()):,}")
    print(f"Predictor parameters: {sum(p.numel() for p in predictor.parameters()):,}")
    
    # Train model
    best_metrics, train_losses, val_losses, val_metrics_history = train(
        train_dataloader, test_dataloader, resnet, predictor, EPOCHS
    )
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    subset_acc = [m['subset_accuracy'] for m in val_metrics_history]
    hamming_acc = [m['hamming_accuracy'] for m in val_metrics_history]
    plt.plot(subset_acc, label='Subset Accuracy')
    plt.plot(hamming_acc, label='Hamming Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    f1_micro = [m['f1_micro'] for m in val_metrics_history]
    f1_macro = [m['f1_macro'] for m in val_metrics_history]
    plt.plot(f1_micro, label='F1 Micro')
    plt.plot(f1_macro, label='F1 Macro')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nBest Model Metrics:")
    print("="*30)
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")