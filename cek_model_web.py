# cek_model_web.py
import pickle
import pandas as pd
import numpy as np

print("="*60)
print("🔍 CEK MODEL YANG DIPAKAI WEB")
print("="*60)

# Load model
try:
    with open('models/model_terbaik.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model berhasil di-load\n")
except:
    print("❌ Model tidak ditemukan di models/model_terbaik.pkl")
    exit()

# 1. Cek jumlah fitur
print(f"📊 1. Jumlah fitur yang digunakan: {model.n_features_in_}")

# 2. Cek nama-nama fitur
try:
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
        print(f"\n📋 2. Nama fitur yang digunakan ({len(feature_names)} fitur):")
        for i, f in enumerate(feature_names, 1):
            print(f"   {i}. {f}")
    else:
        print("\n⚠️ Model tidak menyimpan nama fitur")
except Exception as e:
    print(f"Error: {e}")

# 3. Cek parameter model
print(f"\n⚙️ 3. Parameter Model:")
print(f"   - n_estimators: {model.n_estimators}")
print(f"   - max_depth: {model.max_depth}")
print(f"   - random_state: {model.random_state}")

# 4. Cek jenis model
print(f"\n🏷️ 4. Jenis Model: {type(model).__name__}")

# 5. Cek ukuran model (berapa banyak decision tree)
if hasattr(model, 'estimators_'):
    print(f"   - Jumlah decision tree: {len(model.estimators_)}")

print("\n" + "="*60)

# 6. Kesimpulan berdasarkan jumlah fitur
print("\n🎯 KESIMPULAN:")
print("="*60)

if model.n_features_in_ == 5:
    print("✅ Model menggunakan 5 FITUR FISIK SAJA")
    print("   Fitur: kamar_tidur, kamar_mandi, luas_tanah_m2, luas_bangunan, kecamatan")
    print("   → Ini adalah MODEL 1 (Fisik saja)")

elif model.n_features_in_ == 6:
    print("✅ Model menggunakan 6 FITUR")
    print("   Fitur: fisik + kurs_usd")
    print("   → Ini adalah MODEL 2 (+ Kurs)")

elif model.n_features_in_ == 7:
    print("✅ Model menggunakan 7 FITUR")
    print("   Fitur: fisik + kurs_usd + inflasi_malang")
    print("   → Ini adalah MODEL 3 (+ Kurs + Inflasi Malang)")

elif model.n_features_in_ == 8:
    print("✅ Model menggunakan 8 FITUR")
    print("   Fitur: fisik + kurs_usd + inflasi_malang + inflasi_nasional")
    print("   → Ini adalah MODEL 4 (+ Semua Makro)")

elif model.n_features_in_ == 10:
    print("✅ Model menggunakan 10 FITUR")
    print("   Fitur: fisik + kurs_usd + inflasi_malang + inflasi_nasional + ihk_umum + ihk_perumahan")
    print("   → Ini adalah MODEL 5 (+ Semua Makro + IHK)")

else:
    print(f"⚠️ Model menggunakan {model.n_features_in_} fitur (tidak standar)")

print("="*60)