import torch
import torchvision
from torch import nn
from torch.nn import functional as F
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

# Loss Functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # Probability
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        # Clipping
        if self.clip > 0:
            xs_neg = torch.clamp(xs_neg, min=self.clip)

        # Loss calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric focusing
        loss = los_pos * torch.pow(1 - xs_pos, self.gamma_pos) + \
               los_neg * torch.pow(xs_pos, self.gamma_neg)
        
        return -loss.mean()

class ClassBalancedLoss(nn.Module):
    def __init__(self, num_classes, beta=0.9999):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        
    def forward(self, logits, labels, class_counts):
        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(self.beta, class_counts.float())
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * self.num_classes
        
        # Apply weights to BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        weighted_loss = bce_loss * weights.unsqueeze(0).to(logits.device)
        
        return weighted_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, loss_type='focal', alpha=0.7, beta=0.3, **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        if loss_type == 'focal':
            self.secondary_loss = FocalLoss(**kwargs)
        elif loss_type == 'asymmetric':
            self.secondary_loss = AsymmetricLoss(**kwargs)
        elif loss_type == 'class_balanced':
            self.secondary_loss = ClassBalancedLoss(**kwargs)
        
    def forward(self, logits, labels, **kwargs):
        primary_loss = self.bce_loss(logits, labels)
        
        if self.loss_type == 'class_balanced':
            secondary_loss = self.secondary_loss(logits, labels, **kwargs)
        else:
            secondary_loss = self.secondary_loss(logits, labels)
        
        return self.alpha * primary_loss + self.beta * secondary_loss

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
        return image, label, name  # Return name for debugging if needed

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

def evaluate_model(model, predictor, dataloader, criterion, device=DEVICE):
    """Comprehensive model evaluation"""
    model.eval()
    predictor.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    total_loss = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating"):
            if len(batch_data) == 3:
                images, labels, _ = batch_data  # Unpack with names
            else:
                images, labels = batch_data
            
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

def train(train_dataloader, test_dataloader, model, predictor, epochs, 
          loss_type='focal', device=DEVICE):
    """
    Training function with configurable loss functions
    
    Args:
        loss_type: 'bce', 'focal', 'asymmetric', 'class_balanced', 'combined'
    """
    
    optimizer = AdamW([
        {'params': predictor.parameters(), 'lr': 1e-5},
        {'params': model.parameters(), 'lr': 1e-6}
    ])
    
    # Choose loss function based on loss_type
    if loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)
    elif loss_type == 'asymmetric':
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    elif loss_type == 'class_balanced':
        # Calculate class frequencies first
        class_counts = torch.zeros(100)
        for batch_data in train_dataloader:
            if len(batch_data) == 3:
                _, labels, _ = batch_data
            else:
                _, labels = batch_data
            class_counts += labels.sum(dim=0)
        criterion = ClassBalancedLoss(num_classes=100, beta=0.9999)
    elif loss_type == 'combined':
        criterion = CombinedLoss(loss_type='focal', alpha=0.7, beta=0.3)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    print(f"Using loss function: {loss_type}")
    
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
        for batch_data in loop:
            if len(batch_data) == 3:
                images, labels, _ = batch_data
            else:
                images, labels = batch_data
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            features = model(images)
            pred = predictor(features)
            
            # Calculate loss based on type
            if loss_type == 'class_balanced':
                loss = criterion(pred, labels, class_counts)
            else:
                loss = criterion(pred, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_metrics, val_labels, val_preds, val_scores = evaluate_model(
            model, predictor, test_dataloader, criterion, device
        )
        
        val_losses.append(val_metrics['avg_loss'])
        val_metrics_history.append(val_metrics)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs} - Loss Type: {loss_type}")
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
                'train_loss': avg_train_loss,
                'loss_type': loss_type
            }
            torch.save(checkpoint, os.path.join(model_output_dir, f'best_model_{loss_type}_epoch_{epoch+1}.pt'))
            
            # Save detailed evaluation for best model
            if epoch == epochs - 1 or epoch % 10 == 0:  # Save detailed metrics periodically
                # Plot confusion matrices
                plot_confusion_matrices(
                    val_labels, val_preds, 100, 
                    save_path=os.path.join(model_output_dir, f'confusion_matrices_{loss_type}_epoch_{epoch+1}.png')
                )
                
                # Plot label distribution
                plot_label_distribution(
                    val_labels, val_preds,
                    save_path=os.path.join(model_output_dir, f'label_distribution_{loss_type}_epoch_{epoch+1}.png')
                )
                
                # Save detailed classification report
                with open(os.path.join(model_output_dir, f'classification_report_{loss_type}_epoch_{epoch+1}.txt'), 'w') as f:
                    f.write(f"Multi-label Classification Report - Loss: {loss_type}\n")
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
    
    # Test different loss functions
    loss_functions = ['bce', 'focal', 'asymmetric']  # Add more as needed
    
    results = {}
    
    for loss_type in loss_functions:
        print(f"\n{'='*60}")
        print(f"TRAINING WITH {loss_type.upper()} LOSS")
        print(f"{'='*60}")
        
        # Reinitialize model for fair comparison
        resnet = models.resnet50(weights='DEFAULT')
        resnet.fc = nn.Identity()
        resnet = resnet.to(DEVICE)
        predictor = classifier(num_classes=100)
        
        # Train model with specific loss function
        best_metrics, train_losses, val_losses, val_metrics_history = train(
            train_dataloader, test_dataloader, resnet, predictor, 
            epochs=20,  # Reduced for testing multiple loss functions
            loss_type=loss_type
        )
        
        # Store results
        results[loss_type] = {
            'best_metrics': best_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics_history': val_metrics_history
        }
        
        print(f"\nBest {loss_type.upper()} Results:")
        print(f"Subset Accuracy: {best_metrics['subset_accuracy']:.4f}")
        print(f"Hamming Accuracy: {best_metrics['hamming_accuracy']:.4f}")
        print(f"F1 Micro: {best_metrics['f1_micro']:.4f}")
        print(f"F1 Macro: {best_metrics['f1_macro']:.4f}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON OF LOSS FUNCTIONS")
    print(f"{'='*60}")
    
    comparison_metrics = ['subset_accuracy', 'hamming_accuracy', 'f1_micro', 'f1_macro']
    
    for metric in comparison_metrics:
        print(f"\n{metric.upper()}:")
        for loss_type in loss_functions:
            value = results[loss_type]['best_metrics'][metric]
            print(f"  {loss_type:12}: {value:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(comparison_metrics):
        ax = axes[i]
        values = [results[loss_type]['best_metrics'][metric] for loss_type in loss_functions]
        bars = ax.bar(loss_functions, values, alpha=0.7, color=['blue', 'red', 'green', 'orange'][:len(loss_functions)])
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, 'loss_function_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find best loss function
    best_loss_type = max(loss_functions, key=lambda x: results[x]['best_metrics']['f1_micro'])
    print(f"\nBEST LOSS FUNCTION: {best_loss_type.upper()}")
    print(f"F1 Micro Score: {results[best_loss_type]['best_metrics']['f1_micro']:.4f}")
    
    # Save comparison results
    with open(os.path.join(model_output_dir, 'loss_function_comparison.txt'), 'w') as f:
        f.write("Loss Function Comparison Results\n")
        f.write("="*40 + "\n\n")
        
        for loss_type in loss_functions:
            f.write(f"{loss_type.upper()} LOSS:\n")
            for metric, value in results[loss_type]['best_metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
        
        f.write(f"BEST LOSS FUNCTION: {best_loss_type.upper()}\n")
        f.write(f"F1 Micro Score: {results[best_loss_type]['best_metrics']['f1_micro']:.4f}\n")
    
    # Plot training curves for the best loss function
    plt.figure(figsize=(15, 5))
    
    best_train_losses = results[best_loss_type]['train_losses']
    best_val_losses = results[best_loss_type]['val_losses']
    best_val_metrics_history = results[best_loss_type]['val_metrics_history']
    
    plt.subplot(1, 3, 1)
    plt.plot(best_train_losses, label='Train Loss')
    plt.plot(best_val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Curves - {best_loss_type.upper()}')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    subset_acc = [m['subset_accuracy'] for m in best_val_metrics_history]
    hamming_acc = [m['hamming_accuracy'] for m in best_val_metrics_history]
    plt.plot(subset_acc, label='Subset Accuracy')
    plt.plot(hamming_acc, label='Hamming Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Metrics')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    f1_micro = [m['f1_micro'] for m in best_val_metrics_history]
    f1_macro = [m['f1_macro'] for m in best_val_metrics_history]
    plt.plot(f1_micro, label='F1 Micro')
    plt.plot(f1_macro, label='F1 Macro')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nBest Model ({best_loss_type.upper()}) Metrics:")
    print("="*30)
    for metric, value in results[best_loss_type]['best_metrics'].items():
        print(f"{metric}: {value:.4f}")