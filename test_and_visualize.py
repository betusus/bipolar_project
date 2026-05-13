# -*- coding: utf-8 -*-
"""
Bipolar Projesi — Poster Görselleri
@author: Betul Savas

Üretilen görseller:
  1. confusion_matrix.png   — Karmaşıklık matrisi
  2. roc_curve.png          — ROC eğrisi + AUC
  3. metrics_bar.png        — Metrik karşılaştırma çubuğu
  4. poster_combined.png    — Tüm görseller tek dosyada
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, f1_score,
    precision_score, recall_score
)
import os

# ============================================================
# AYARLAR
# ============================================================
BASE_PATH  = r"C:\Users\Betul Savas\bipolar_proje\processed_v4"
TEST_DIR   = os.path.join(BASE_PATH, "test")
MODEL_PATH = "bipolar_efficientnet_v3_best.pth"
OUTPUT_DIR = "poster_gorseller"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Poster font ayarları
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 13,
    'axes.titlesize'   : 15,
    'axes.titleweight' : 'bold',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
})

# Renk paleti
C_TEAL   = '#1D9E75'   # Doğru tahmin
C_BLUE   = '#185FA5'   # İkincil
C_RED    = '#E24B4A'   # Yanlış tahmin
C_LIGHT  = '#E1F5EE'   # Açık teal
C_GRAY   = '#888780'   # Nötr
C_DARK   = '#2C2C2A'   # Koyu metin

# ============================================================
# MODEL VE VERİ
# ============================================================
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data   = datasets.ImageFolder(TEST_DIR, test_transforms)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
class_names = test_data.classes   # ['BIPOLAR', 'CONTROL']

model = models.efficientnet_b0(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.45),
    nn.Linear(num_ftrs, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        probs    = torch.softmax(outputs, dim=1)[:, 0]
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

cm = confusion_matrix(all_labels, all_preds)
print("Sınıf eşleşmesi:", test_data.class_to_idx)
print(classification_report(all_labels, all_preds, target_names=class_names))

# ============================================================
# 1. CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    fig.patch.set_facecolor('white')

    # Arka plan renkleri: doğru=teal, yanlış=kırmızı
    colors = np.array([
        [C_TEAL, C_RED],
        [C_RED,  C_TEAL]
    ])

    for i in range(2):
        for j in range(2):
            rect = FancyBboxPatch(
                (j - 0.45, i - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=colors[i][j],
                edgecolor='white', linewidth=3,
                transform=ax.transData, zorder=2
            )
            ax.add_patch(rect)

            # Büyük sayı
            ax.text(j, i - 0.05, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=30, fontweight='bold',
                    color='white', zorder=3)

            # Alt etiket
            label = 'Doğru ✓' if i == j else 'Yanlış ✗'
            ax.text(j, i + 0.28, label,
                    ha='center', va='center',
                    fontsize=10, color='white',
                    alpha=0.9, zorder=3)

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=13, fontweight='bold')
    ax.set_yticklabels(class_names, fontsize=13, fontweight='bold')
    ax.set_xlabel('Modelin Tahmini', fontsize=13, labelpad=12)
    ax.set_ylabel('Gerçek Durum', fontsize=13, labelpad=12)
    ax.set_title('Karmaşıklık Matrisi', fontsize=16,
                 fontweight='bold', pad=16, color=C_DARK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Metrik özeti altta
    acc    = np.diag(cm).sum() / cm.sum()
    bp_rcl = cm[0, 0] / cm[0].sum()
    ct_rcl = cm[1, 1] / cm[1].sum()
    fig.text(0.5, -0.02,
             f'Accuracy: %{acc*100:.1f}   |   '
             f'Bipolar Recall: %{bp_rcl*100:.1f}   |   '
             f'Control Recall: %{ct_rcl*100:.1f}',
             ha='center', fontsize=11, color=C_GRAY)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ {save_path}")

plot_confusion_matrix(cm, class_names,
                      os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

# ============================================================
# 2. ROC EĞRİSİ
# ============================================================
def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=0)
    roc_auc  = auc(fpr, tpr)
    best_idx = np.argmax(tpr - fpr)
    best_thr = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(6, 5.2))
    fig.patch.set_facecolor('white')

    # Gölge alan
    ax.fill_between(fpr, tpr, alpha=0.12, color=C_TEAL)

    # ROC eğrisi
    ax.plot(fpr, tpr, color=C_TEAL, lw=2.5,
            label=f'EfficientNet-B0  (AUC = {roc_auc:.3f})')

    # Rastgele sınıflandırıcı
    ax.plot([0, 1], [0, 1], color=C_GRAY, lw=1.5,
            linestyle='--', label='Rastgele sınıflandırıcı')

    # Optimal nokta
    ax.scatter(fpr[best_idx], tpr[best_idx],
               color=C_RED, s=120, zorder=5,
               label=f'Optimal eşik = {best_thr:.3f}')
    ax.annotate(f'  ({fpr[best_idx]:.2f}, {tpr[best_idx]:.2f})',
                xy=(fpr[best_idx], tpr[best_idx]),
                fontsize=10, color=C_RED, va='bottom')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Yanlış Pozitif Oranı (1 - Özgüllük)', fontsize=12)
    ax.set_ylabel('Doğru Pozitif Oranı (Duyarlılık)', fontsize=12)
    ax.set_title('ROC Eğrisi', fontsize=16,
                 fontweight='bold', pad=16, color=C_DARK)
    ax.legend(loc='lower right', fontsize=11,
              framealpha=0.9, edgecolor='#ddd')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ {save_path}")

plot_roc_curve(all_labels, all_probs,
               os.path.join(OUTPUT_DIR, "roc_curve.png"))

# ============================================================
# 3. METRİK ÇUBUĞU
# ============================================================
def plot_metrics_bar(labels, preds, class_names, save_path):
    metric_names = ['Precision', 'Recall', 'F1-Score']
    bp_vals = [
        precision_score(labels, preds, pos_label=0, average='binary'),
        recall_score(labels,    preds, pos_label=0, average='binary'),
        f1_score(labels,        preds, pos_label=0, average='binary'),
    ]
    ct_vals = [
        precision_score(labels, preds, pos_label=1, average='binary'),
        recall_score(labels,    preds, pos_label=1, average='binary'),
        f1_score(labels,        preds, pos_label=1, average='binary'),
    ]

    x     = np.arange(len(metric_names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5.2))
    fig.patch.set_facecolor('white')

    bars1 = ax.bar(x - width/2, bp_vals, width,
                   label='BIPOLAR', color=C_TEAL,
                   alpha=0.88, zorder=3)
    bars2 = ax.bar(x + width/2, ct_vals, width,
                   label='CONTROL', color=C_BLUE,
                   alpha=0.88, zorder=3)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.015, f'{h:.2f}',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=C_DARK)

    # Macro F1 çizgisi
    macro_f1 = f1_score(labels, preds, average='macro')
    ax.axhline(y=macro_f1, color=C_RED, linestyle='--',
               lw=1.8, alpha=0.8, zorder=2)
    ax.text(2.42, macro_f1 + 0.02,
            f'Macro F1: {macro_f1:.2f}',
            ha='right', fontsize=10,
            color=C_RED, fontweight='bold')

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=13)
    ax.set_ylabel('Skor', fontsize=13)
    ax.set_title('Sınıf Bazlı Performans Metrikleri',
                 fontsize=16, fontweight='bold', pad=16, color=C_DARK)
    ax.legend(fontsize=12, framealpha=0.9, edgecolor='#ddd')
    ax.grid(axis='y', alpha=0.25, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor('#FAFAFA')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ {save_path}")

plot_metrics_bar(all_labels, all_preds, class_names,
                 os.path.join(OUTPUT_DIR, "metrics_bar.png"))

# ============================================================
# 4. KOMBİNE POSTER GÖRSELİ (3 grafik yan yana)
# ============================================================
def plot_poster_combined(cm, labels, preds, probs, class_names, save_path):
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(1, 3, wspace=0.38)

    # --- Sol: Confusion Matrix ---
    ax1 = fig.add_subplot(gs[0])
    colors = np.array([[C_TEAL, C_RED], [C_RED, C_TEAL]])
    for i in range(2):
        for j in range(2):
            rect = FancyBboxPatch(
                (j - 0.44, i - 0.44), 0.88, 0.88,
                boxstyle="round,pad=0.04",
                facecolor=colors[i][j],
                edgecolor='white', linewidth=3,
                transform=ax1.transData, zorder=2
            )
            ax1.add_patch(rect)
            ax1.text(j, i - 0.05, str(cm[i, j]),
                     ha='center', va='center',
                     fontsize=28, fontweight='bold',
                     color='white', zorder=3)
            label = 'Doğru ✓' if i == j else 'Yanlış ✗'
            ax1.text(j, i + 0.27, label,
                     ha='center', va='center',
                     fontsize=9, color='white', zorder=3)

    ax1.set_xlim(-0.55, 1.55)
    ax1.set_ylim(-0.55, 1.55)
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(class_names, fontsize=12, fontweight='bold')
    ax1.set_yticklabels(class_names, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Modelin Tahmini', fontsize=12, labelpad=10)
    ax1.set_ylabel('Gerçek Durum', fontsize=12, labelpad=10)
    ax1.set_title('Karmaşıklık Matrisi', fontsize=14,
                  fontweight='bold', pad=14, color=C_DARK)
    ax1.tick_params(length=0)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # --- Orta: ROC ---
    ax2 = fig.add_subplot(gs[1])
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=0)
    roc_auc  = auc(fpr, tpr)
    best_idx = np.argmax(tpr - fpr)

    ax2.fill_between(fpr, tpr, alpha=0.12, color=C_TEAL)
    ax2.plot(fpr, tpr, color=C_TEAL, lw=2.5,
             label=f'AUC = {roc_auc:.3f}')
    ax2.plot([0, 1], [0, 1], color=C_GRAY, lw=1.5, linestyle='--',
             label='Rastgele')
    ax2.scatter(fpr[best_idx], tpr[best_idx],
                color=C_RED, s=100, zorder=5,
                label=f'Eşik = {thresholds[best_idx]:.3f}')
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])
    ax2.set_xlabel('Yanlış Pozitif Oranı', fontsize=12)
    ax2.set_ylabel('Doğru Pozitif Oranı', fontsize=12)
    ax2.set_title('ROC Eğrisi', fontsize=14,
                  fontweight='bold', pad=14, color=C_DARK)
    ax2.legend(loc='lower right', fontsize=10,
               framealpha=0.9, edgecolor='#ddd')
    ax2.grid(True, alpha=0.25, linestyle='--')
    ax2.set_facecolor('#FAFAFA')

    # --- Sağ: Metrik çubuğu ---
    ax3 = fig.add_subplot(gs[2])
    metric_names = ['Precision', 'Recall', 'F1']
    bp_vals = [
        precision_score(labels, preds, pos_label=0, average='binary'),
        recall_score(labels,    preds, pos_label=0, average='binary'),
        f1_score(labels,        preds, pos_label=0, average='binary'),
    ]
    ct_vals = [
        precision_score(labels, preds, pos_label=1, average='binary'),
        recall_score(labels,    preds, pos_label=1, average='binary'),
        f1_score(labels,        preds, pos_label=1, average='binary'),
    ]
    x     = np.arange(len(metric_names))
    width = 0.32
    b1 = ax3.bar(x - width/2, bp_vals, width,
                 label='BIPOLAR', color=C_TEAL, alpha=0.88, zorder=3)
    b2 = ax3.bar(x + width/2, ct_vals, width,
                 label='CONTROL', color=C_BLUE, alpha=0.88, zorder=3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 h + 0.015, f'{h:.2f}',
                 ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color=C_DARK)
    macro_f1 = f1_score(labels, preds, average='macro')
    ax3.axhline(y=macro_f1, color=C_RED, linestyle='--',
                lw=1.8, alpha=0.8, zorder=2)
    ax3.text(2.42, macro_f1 + 0.02,
             f'Macro F1\n{macro_f1:.2f}',
             ha='right', fontsize=9,
             color=C_RED, fontweight='bold')
    ax3.set_ylim(0, 1.12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names, fontsize=12)
    ax3.set_ylabel('Skor', fontsize=12)
    ax3.set_title('Performans Metrikleri', fontsize=14,
                  fontweight='bold', pad=14, color=C_DARK)
    ax3.legend(fontsize=11, framealpha=0.9, edgecolor='#ddd')
    ax3.grid(axis='y', alpha=0.25, linestyle='--', zorder=0)
    ax3.set_axisbelow(True)
    ax3.set_facecolor('#FAFAFA')
    for spine in ['top', 'right']:
        ax3.spines[spine].set_visible(False)

    # Genel başlık
    fig.suptitle(
        'EfficientNet-B0 — MRI Görüntülerinden Bipolar Bozukluk Tespiti',
        fontsize=17, fontweight='bold', y=1.03, color=C_DARK
    )

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ {save_path}")

plot_poster_combined(
    cm, all_labels, all_preds, all_probs, class_names,
    os.path.join(OUTPUT_DIR, "poster_combined.png")
)

print(f"\n{'='*55}")
print(f"Tüm görseller '{OUTPUT_DIR}/' klasörüne kaydedildi.")
print(f"  confusion_matrix.png  — tek başına (poster sol)")
print(f"  roc_curve.png         — tek başına (poster orta)")
print(f"  metrics_bar.png       — tek başına (poster sağ)")
print(f"  poster_combined.png   — üçü yan yana tek dosya")
print(f"{'='*55}")
print(f"\nModel: {MODEL_PATH}")
print(f"Accuracy  : %{100*np.diag(cm).sum()/cm.sum():.1f}")
print(f"BP Recall : %{100*cm[0,0]/cm[0].sum():.1f}")
print(f"Macro F1  : {f1_score(all_labels, all_preds, average='macro'):.4f}")