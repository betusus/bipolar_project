import os
import random
import shutil

# --- YOLLAR ---
BASE_PATH  = r"C:\Users\Betul Savas\bipolar_proje\processed_v3"
SOURCE_DIR = os.path.join(BASE_PATH, "all_images")
TRAIN_DIR  = os.path.join(BASE_PATH, "train")
TEST_DIR   = os.path.join(BASE_PATH, "test")

def split_data_patient_wise(test_ratio=0.2, seed=42):
    random.seed(seed)

    # 1. Klasör yapısını hazırla
    for phase in [TRAIN_DIR, TEST_DIR]:
        for label in ["BIPOLAR", "CONTROL"]:
            os.makedirs(os.path.join(phase, label), exist_ok=True)

    # 2. Bütün dosyaları listele
    if not os.path.exists(SOURCE_DIR):
        print(f"HATA: Kaynak klasör bulunamadı: {SOURCE_DIR}")
        return

    all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')]

    # 3. Hastaları sınıflarına göre grupla
    subjects = {"BIPOLAR": set(), "CONTROL": set()}
    
    for f in all_files:
        # Örn: BIPOLAR_sub-60011_slice17.png -> split('_')
        # parts[0] = "BIPOLAR"
        # parts[1] = "sub-60011"
        parts = f.split('_') 
        
        if len(parts) < 2:
            continue
            
        label = parts[0].upper() 
        subject_id = parts[1] # "sub-60011" tam olarak alınır
        
        if label in subjects:
            subjects[label].add(subject_id)

    print(f"Tespit edilen toplam hastalar: "
          f"Bipolar: {len(subjects['BIPOLAR'])}, Control: {len(subjects['CONTROL'])}")

    # 4. Her sınıfı kendi içinde böl
    for label in ["BIPOLAR", "CONTROL"]:
        sub_list = list(subjects[label])
        random.shuffle(sub_list)

        test_count    = int(len(sub_list) * test_ratio)
        test_subjects = set(sub_list[:test_count])

        print(f"\n{label} bölme detayları:")
        print(f"- Toplam hasta: {len(sub_list)}")
        print(f"- Eğitime ayrılan: {len(sub_list) - test_count} hasta")
        print(f"- Teste ayrılan: {test_count} hasta")

        # 5. Dosyaları kopyala
        count = 0
        for f in all_files:
            f_parts = f.split('_')
            f_label = f_parts[0].upper()
            f_subject = f_parts[1]
            
            if f_label != label:
                continue
                
            # Eğer bu hasta test grubundaysa TEST_DIR'e, değilse TRAIN_DIR'e
            target_phase = TEST_DIR if f_subject in test_subjects else TRAIN_DIR
            shutil.copy(
                os.path.join(SOURCE_DIR, f),
                os.path.join(target_phase, label, f)
            )
            count += 1
        print(f"- Toplam {count} adet kesit kopyalandı.")

    print("\n--- İŞLEM BAŞARIYLA TAMAMLANDI ---")

if __name__ == "__main__":
    split_data_patient_wise(test_ratio=0.2)