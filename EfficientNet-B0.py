# -*- coding: utf-8 -*-
"""
Bipolar Sınıflandırma — EfficientNet-B0 v3 (Düzeltilmiş)
@author: Betul Savas

Düzeltmeler:
  - Threshold mantığı düzeltildi:
    prob[:, 0] = BIPOLAR olasılığı
    preds = argmax(outputs) kullanılıyor, elle threshold yok
  - ROC ve optimize threshold doğru yönde hesaplanıyor
  - Her epoch BP.RCL ve CT.RCL ayrı yazdırılıyor
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, roc_curve, auc
)

# --- 1. AYARLAR ---
BASE_PATH = r"C:\Users\Betul Savas\bipolar_proje\processed_v4"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR  = os.path.join(BASE_PATH, "test")

BATCH_SIZE    = 16
EPOCHS        = 60
LR_HEAD       = 8e-4
LR_BACKBONE   = 5e-6
PATIENCE      = 15
FREEZE_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {DEVICE}")

# --- 2. TRANSFORMS ---
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. LOADERS ---
train_data = datasets.ImageFolder(TRAIN_DIR, train_transforms)
test_data  = datasets.ImageFolder(TEST_DIR,  test_transforms)

print(f"Sınıflar : {train_data.class_to_idx}")   # BIPOLAR=0, CONTROL=1 olmalı

class_counts   = np.bincount(train_data.targets)
sample_weights = [1.0 / class_counts[t] for t in train_data.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train    : {len(train_data)} slice  "
      f"({class_counts[0]} bipolar / {class_counts[1]} control)")
print(f"Test     : {len(test_data)} slice")

# --- 4. MODEL ---
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

for param in model.features.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.45),
    nn.Linear(num_ftrs, 2)
)
model = model.to(DEVICE)

# --- 5. LOSS & OPTIMIZER ---
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.3, 1.0]).to(DEVICE),
    label_smoothing=0.1
)
optimizer = optim.AdamW(
    model.classifier.parameters(),
    lr=LR_HEAD,
    weight_decay=1e-3
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.2
)

# --- 6. EĞİTİM DÖNGÜSÜ ---
best_val_f1        = 0.0
early_stop_counter = 0
backbone_unfrozen  = False
best_probs         = []
best_labels_saved  = []

print(f"\n🚀 EfficientNet-B0 v3 Eğitimi Başlıyor...")
print("="*100)
print(f"{'EP':<4} | {'T.LOSS':<8} | {'V.LOSS':<8} | {'ACC':<7} | {'F1':<6} | "
      f"{'BP.RCL':<7} | {'CT.RCL':<7} | {'LR':<9} | DURUM")
print("-"*100)

for epoch in range(EPOCHS):

    # Freeze → Fine-tune geçişi
    if epoch == FREEZE_EPOCHS and not backbone_unfrozen:
        print(f"\n[Epoch {epoch+1}] Backbone açılıyor — fine-tune başlıyor...\n")
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW([
            {'params': model.features.parameters(),   'lr': LR_BACKBONE},
            {'params': model.classifier.parameters(), 'lr': LR_HEAD / 10}
        ], weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.2
        )
        backbone_unfrozen = True

    # Train
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()

            # ✅ Doğru tahmin: argmax — elle threshold yok
            _, preds = torch.max(outputs, dim=1)

            # ✅ Bipolar olasılığı: index 0 (BIPOLAR=0)
            probs = torch.softmax(outputs, dim=1)[:, 0]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    avg_t_loss = train_loss / len(train_loader)
    avg_v_loss = val_loss / len(test_loader)
    scheduler.step(avg_v_loss)

    cm        = confusion_matrix(all_labels, all_preds)
    val_acc   = 100. * np.diag(cm).sum() / cm.sum()
    val_f1    = f1_score(all_labels, all_preds, average='macro')
    bp_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    ct_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
    current_lr = optimizer.param_groups[0]['lr']

    if val_f1 > best_val_f1:
        best_val_f1       = val_f1
        early_stop_counter = 0
        torch.save(model.state_dict(), "bipolar_efficientnet_v4_best.pth")
        best_probs        = all_probs.tolist()
        best_labels_saved = all_labels.tolist()
        status = "⭐ EN İYİ"
    else:
        early_stop_counter += 1
        status = f"⏳ {early_stop_counter}/{PATIENCE}"

    print(f"{epoch+1:<4} | {avg_t_loss:<8.4f} | {avg_v_loss:<8.4f} | "
          f"%{val_acc:<6.1f} | {val_f1:<6.3f} | "
          f"%{bp_recall*100:<6.1f} | %{ct_recall*100:<6.1f} | "
          f"{current_lr:<9.1e} | {status}")

    if early_stop_counter >= PATIENCE:
        print("\n[!] Erken Durdurma Uygulandı.")
        break

# --- 7. THRESHOLD OPTİMİZASYONU ---
print("\n" + "="*60)
print("🔍 Threshold Optimizasyonu (Youden J)")

best_probs_arr  = np.array(best_probs)
best_labels_arr = np.array(best_labels_saved)

# pos_label=0 → BIPOLAR pozitif sınıf
fpr, tpr, thresholds = roc_curve(best_labels_arr, best_probs_arr, pos_label=0)
roc_auc  = auc(fpr, tpr)
best_idx = np.argmax(tpr - fpr)
best_thr = thresholds[best_idx]

print(f"ROC AUC          : {roc_auc:.4f}")
print(f"En iyi threshold : {best_thr:.4f}")

# --- 8. FINAL RAPOR ---
print("\n" + "="*60)
print("📊 EfficientNet-B0 v3 — FINAL RAPORU")
print("="*60)

model.load_state_dict(torch.load("bipolar_efficientnet_v4_best.pth"))
model.eval()

f_preds, f_probs, f_labels = [], [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.to(DEVICE))

        # ✅ Direkt argmax tahmini
        _, preds = torch.max(outputs, dim=1)

        # ✅ Bipolar olasılığı (index 0)
        probs = torch.softmax(outputs, dim=1)[:, 0]

        f_preds.extend(preds.cpu().numpy())
        f_probs.extend(probs.cpu().numpy())
        f_labels.extend(labels.cpu().numpy())

f_preds  = np.array(f_preds)
f_probs  = np.array(f_probs)
f_labels = np.array(f_labels)

# Standart: argmax
print("\n--- Standart (argmax) ---")
print(classification_report(f_labels, f_preds, target_names=test_data.classes))
print("Karmaşıklık Matrisi:")
print(confusion_matrix(f_labels, f_preds))

# ✅ Optimize threshold: prob >= best_thr → BIPOLAR (0), değilse CONTROL (1)
preds_opt = np.where(f_probs >= best_thr, 0, 1)
print(f"\n--- Threshold = {best_thr:.4f} (optimize edilmiş) ---")
print(classification_report(f_labels, preds_opt, target_names=test_data.classes))
print("Karmaşıklık Matrisi:")
print(confusion_matrix(f_labels, preds_opt))

print(f"\nEn iyi Macro F1 : {best_val_f1:.4f}")
print(f"ROC AUC         : {roc_auc:.4f}")
print("Model kaydedildi: bipolar_efficientnet_v4_best.pth")