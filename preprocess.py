# -*- coding: utf-8 -*-
"""
Bipolar Projesi — Görüntü Ön İşleme
v3 (ham) → v4 (işlenmiş)

İyileştirmeler:
  1. Grayscale → BGR dönüşümü eklendi (EfficientNet 3 kanal bekliyor)
  2. Skull stripping eklendi (beyin dışı dokuları temizle)
  3. CLAHE clipLimit hafifletildi (2.0 → 1.5)
  4. Görüntü 224x224'e preprocess aşamasında boyutlandırıldı
  5. Boyut ve içerik kontrolü eklendi
"""

import cv2
import os
import numpy as np
from tqdm import tqdm

# --- YOLLAR ---
INPUT_BASE  = r"C:\Users\Betul Savas\bipolar_proje\processed_v3"
OUTPUT_BASE = r"C:\Users\Betul Savas\bipolar_proje\processed_v4"

# CLAHE — clipLimit hafifletildi, T1 MRI için daha koruyucu
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

def skull_strip(img):
    """
    Kaba skull stripping — kafa derisi ve kafatasını maskeler,
    sadece beyin bölgesini bırakır.
    Gerçek FSL/FreeSurfer kadar hassas değil ama
    derin öğrenme için yeterli temizliği sağlar.
    """
    # Eşikleme ile beyin maskesi oluştur
    _, mask = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)

    # Morfolojik kapatma — küçük delikleri doldur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

    # En büyük bağlı bileşeni al (beyin kütlesi)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # En büyük bileşen (index 0 arka plan)
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.uint8(labels == largest) * 255

    # Maskeyi uygula
    return cv2.bitwise_and(img, mask)


def process_and_copy():
    print("🚀 Ön İşleme Başlatıldı: v3 → v4")
    print(f"   Giriş : {INPUT_BASE}")
    print(f"   Çıkış : {OUTPUT_BASE}\n")

    if not os.path.exists(OUTPUT_BASE):
        os.makedirs(OUTPUT_BASE)

    total_ok      = 0
    total_skipped = 0

    for root, dirs, files in os.walk(INPUT_BASE):
        relative_path = os.path.relpath(root, INPUT_BASE)
        target_path   = os.path.join(OUTPUT_BASE, relative_path)
        os.makedirs(target_path, exist_ok=True)

        image_files = [f for f in files if f.endswith('.png')]
        if not image_files:
            continue

        for filename in tqdm(image_files, desc=f"{relative_path}", leave=False):
            img_path = os.path.join(root, filename)
            out_path = os.path.join(target_path, filename)

            # 1. Grayscale oku
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Hata kontrolü
            if img is None or img.size == 0:
                print(f"  ⚠ Okunamadı: {filename}")
                total_skipped += 1
                continue
            if img.shape[0] < 50 or img.shape[1] < 50:
                print(f"  ⚠ Çok küçük ({img.shape}): {filename}")
                total_skipped += 1
                continue

            # 2. Gürültü giderme (Median Blur)
            denoised = cv2.medianBlur(img, 3)

            # 3. Skull stripping — beyin dışını maskele
            brain_only = skull_strip(denoised)

            # 4. CLAHE — kontrast artırma
            enhanced = clahe.apply(brain_only)

            # 5. Boyutlandırma — eğitimde her seferinde hesaplanmasın
            resized = cv2.resize(enhanced, (224, 224), interpolation=cv2.INTER_LANCZOS4)

            # 6. ✅ Grayscale → BGR (3 kanal)
            # EfficientNet/DenseNet/ResNet ImageNet ağırlıkları 3 kanal bekliyor
            final_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(out_path, final_img)
            total_ok += 1

    print(f"\n✅ Tamamlandı!")
    print(f"   İşlenen : {total_ok} görüntü")
    print(f"   Atlanan : {total_skipped} görüntü")


if __name__ == "__main__":
    process_and_copy()