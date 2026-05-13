# 🧠 MRI Görüntülerinden Bipolar Bozukluk Tespiti

Pamukkale Üniversitesi — Betül Savaş (22253080)
Danışman: Doç. Dr. Meriç Çetin

## Proje Hakkında
T1-ağırlıklı yapısal MRI görüntülerinden EfficientNet-B0 
derin öğrenme mimarisi kullanılarak Bipolar Bozukluk 
sınıflandırması yapılmıştır.

## Sonuçlar
| Metrik         | Değer  |
|----------------|--------|
| Accuracy       | %66    |
| Bipolar Recall | %80    |
| Macro F1       | 0.66   |
| ROC AUC        | 0.71   |

## Veri Seti
OpenNeuro ds000030
https://openneuro.org/datasets/ds000030/versions/1.0.0

44 Bipolar hasta, 71 sağlıklı kontrol
Her bireyden 12 T1-ağırlıklı MRI slice

## Kurulum
pip install -r requirements.txt

## Kullanım
# 1. Ön işleme
python preprocess.py

# 2. Train/Test bölme
python split.py

# 3. Eğitim
python train_efficientnet_v3.py

# 4. Görsel üretimi
python plot_results.py

## Kullanılan Teknolojiler
- PyTorch
- EfficientNet-B0 (ImageNet ön eğitimli)
- OpenCV (CLAHE, Skull Stripping)
- scikit-learn
