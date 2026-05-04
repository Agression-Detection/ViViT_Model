import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from train import sliding_windows
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
import os
import json

def evaluate(model, test_loader, criterion, device, is_dist, window_size=10, stride=5):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0

    #pbar = tqdm(test_loader, desc="Testing", dynamic_ncols=True)

    with torch.no_grad():
        for videos, labels in test_loader:
            labels = labels.to(device)
            batch_video_logits = []

            for video in videos:
                video = video.to(device)
                if video.shape[0] == 3:
                    video = video.permute(1, 0, 2, 3)
                windows = sliding_windows(
                    video,
                    window_size=window_size,
                    stride=stride
                ).to(device)

                with autocast(device_type="cuda"):
                    outputs = model(windows).logits
                    video_logits = torch.logsumexp(outputs, dim=0)
                batch_video_logits.append(video_logits)

            batch_video_logits = torch.stack(batch_video_logits)
            loss = criterion(batch_video_logits, labels)
            running_loss += loss.item()

            probs = torch.softmax(batch_video_logits, dim=-1)[:, 1]
            preds = torch.argmax(batch_video_logits, dim=-1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = running_loss / len(test_loader)
    return all_preds, all_labels, all_probs, avg_loss

def generate_report(
    all_preds, all_labels, all_probs,
    train_losses, val_losses, val_accs,
    output_dir, model_dir
):
    os.makedirs(output_dir, exist_ok=True)
    class_names = ['non_violent', 'violent']

    # ── Metrics ──────────────────────────────────────────────────────────────
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    roc_auc = roc_auc_score(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    metrics = {
        'accuracy':       report['accuracy'],
        'precision':      report['violent']['precision'],
        'recall':         report['violent']['recall'],
        'f1':             report['violent']['f1-score'],
        'specificity':    specificity,
        'roc_auc':        roc_auc,
        'avg_precision':  avg_precision,
        'support':        int(report['violent']['support']),
    }

    print("\n========== FINAL TEST RESULTS ==========")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    print("========================================\n")

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('ViViT Violence Detection — Training Report', fontsize=16, fontweight='bold')

    # 1. Training curves
    ax1 = fig.add_subplot(2, 3, 1)
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, val_losses,   'r-o', label='Val Loss',   markersize=4)
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 2. Validation accuracy
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(epochs, val_accs, 'g-o', markersize=4)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)

    # 3. Confusion matrix
    ax3 = fig.add_subplot(2, 3, 3)
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title('Confusion Matrix')
    ax3.set_xticks([0, 1]); ax3.set_xticklabels(class_names, rotation=45)
    ax3.set_yticks([0, 1]); ax3.set_yticklabels(class_names)
    plt.colorbar(im, ax=ax3)
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')

    # 4. ROC curve
    ax4 = fig.add_subplot(2, 3, 4)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    ax4.plot(fpr, tpr, 'b-', label=f'ROC (AUC={roc_auc:.3f})')
    ax4.plot([0, 1], [0, 1], 'k--')
    ax4.set_title('ROC Curve')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.legend()
    ax4.grid(True)

    # 5. Precision-Recall curve
    ax5 = fig.add_subplot(2, 3, 5)
    prec, rec, _ = precision_recall_curve(all_labels, all_probs)
    ax5.plot(rec, prec, 'r-', label=f'AP={avg_precision:.3f}')
    ax5.set_title('Precision-Recall Curve')
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.legend()
    ax5.grid(True)

    # 6. Metrics summary table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    table_data = [[k, f'{v:.4f}' if isinstance(v, float) else str(v)]
                  for k, v in metrics.items()]
    table = ax6.table(cellText=table_data, colLabels=['Metric', 'Value'],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax6.set_title('Final Metrics', pad=20)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_report.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Report saved to {plot_path}")
