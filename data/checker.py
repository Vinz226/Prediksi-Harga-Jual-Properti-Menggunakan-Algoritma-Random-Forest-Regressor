import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("🔍 R² DROP CHECKER - ANALISIS PERBANDINGAN DATA")
print("="*80)

# ============================================================
# 1. LOAD BOTH DATASETS
# ============================================================
df_raw = pd.read_csv('rumah123_cleans_final1.csv')  # data raw (R² tinggi)
df_clean = pd.read_csv('rumah123_cleans_final.csv')  # data clean (R² turun)

print("\n📊 PERBANDINGAN UKURAN DATA:")
print(f"   Raw data   : {len(df_raw)} baris")
print(f"   Clean data : {len(df_clean)} baris")

# ============================================================
# 2. CEK DISTRIBUSI HARGA
# ============================================================
print("\n" + "="*60)
print("📈 1. CEK DISTRIBUSI HARGA")
print("="*60)

for name, df in [('Raw', df_raw), ('Clean', df_clean)]:
    if 'harga_bersih' in df.columns:
        print(f"\n   {name} Data:")
        print(f"   Min Harga  : Rp {df['harga_bersih'].min():,.0f}")
        print(f"   Max Harga  : Rp {df['harga_bersih'].max():,.0f}")
        print(f"   Mean Harga : Rp {df['harga_bersih'].mean():,.0f}")
        print(f"   Std Harga  : Rp {df['harga_bersih'].std():,.0f}")
        cv = df['harga_bersih'].std()/df['harga_bersih'].mean() if df['harga_bersih'].mean() > 0 else 0
        print(f"   CV (std/mean): {cv:.2f}")

# ============================================================
# 3. CEK FITUR YANG TERSEDIA
# ============================================================
print("\n" + "="*60)
print("📋 2. CEK KOLOM YANG TERSEDIA")
print("="*60)

print("\n   Raw Data columns:")
for col in df_raw.columns:
    print(f"      - {col}")

print("\n   Clean Data columns:")
for col in df_clean.columns:
    print(f"      - {col}")

# ============================================================
# 4. CEK MISSING VALUES
# ============================================================
print("\n" + "="*60)
print("❓ 3. CEK MISSING VALUES")
print("="*60)

for name, df in [('Raw', df_raw), ('Clean', df_clean)]:
    print(f"\n   {name} Data:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col in missing[missing > 0].index:
        print(f"      {col}: {missing[col]} missing ({missing_pct[col]:.1f}%)")

# ============================================================
# 5. CEK OUTLIER DENGAN IQR
# ============================================================
print("\n" + "="*60)
print("🎯 4. CEK OUTLIER (IQR METHOD)")
print("="*60)

def count_outliers(df, col):
    if col not in df.columns:
        return 0
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return len(df[(df[col] < lower) | (df[col] > upper)])

for name, df in [('Raw', df_raw), ('Clean', df_clean)]:
    print(f"\n   {name} Data:")
    for col in ['harga_bersih', 'luas_tanah_m2', 'kamar_tidur']:
        if col in df.columns:
            outliers = count_outliers(df, col)
            pct = outliers/len(df)*100 if len(df) > 0 else 0
            print(f"      {col}: {outliers} outliers ({pct:.1f}%)")

# ============================================================
# 6. CEK DUPLIKAT
# ============================================================
print("\n" + "="*60)
print("🔄 5. CEK DUPLIKAT")
print("="*60)

for name, df in [('Raw', df_raw), ('Clean', df_clean)]:
    dups = df.duplicated().sum()
    pct = dups/len(df)*100 if len(df) > 0 else 0
    print(f"   {name} Data: {dups} duplikat ({pct:.1f}%)")

# ============================================================
# 7. CEK KORELASI (HANYA UNTUK DATA YANG LENGKAP)
# ============================================================
print("\n" + "="*60)
print("📊 6. CEK KORELASI DENGAN HARGA")
print("="*60)

fitur_fisik = ['kamar_tidur', 'kamar_mandi', 'luas_tanah_m2', 'luas_bangunan']

for name, df in [('Raw', df_raw), ('Clean', df_clean)]:
    print(f"\n   {name} Data:")
    for col in fitur_fisik:
        if col in df.columns and 'harga_bersih' in df.columns:
            # Hapus missing values untuk korelasi
            df_temp = df[[col, 'harga_bersih']].dropna()
            if len(df_temp) > 1:
                corr = df_temp[col].corr(df_temp['harga_bersih'])
                print(f"      {col} vs harga: {corr:.4f}")
            else:
                print(f"      {col} vs harga: (insufficient data)")

# ============================================================
# 8. SIMULASI MODEL UNTUK LIHAT PERFORMA
# ============================================================
print("\n" + "="*60)
print("🤖 7. SIMULASI MODEL (Random Forest)")
print("="*60)

results = []

for name, df in [('Raw', df_raw), ('Clean', df_clean)]:
    print(f"\n   Training on {name} data...")
    
    # Siapkan fitur
    fitur_tersedia = [f for f in fitur_fisik if f in df.columns]
    
    if 'kecamatan' in df.columns:
        fitur_tersedia.append('kecamatan')
    
    if len(fitur_tersedia) < 2:
        print(f"      ⚠️ Fitur tidak cukup untuk training")
        results.append({'dataset': name, 'r2': 0, 'reason': 'Fitur tidak cukup'})
        continue
    
    # Encoding untuk kecamatan
    df_temp = df.copy()
    if 'kecamatan' in df_temp.columns:
        le = LabelEncoder()
        df_temp['kecamatan'] = le.fit_transform(df_temp['kecamatan'].astype(str))
    
    # Hapus missing values
    df_model = df_temp[fitur_tersedia + ['harga_bersih']].dropna()
    
    if len(df_model) < 10:
        print(f"      ⚠️ Data terlalu sedikit ({len(df_model)} baris)")
        results.append({'dataset': name, 'r2': 0, 'reason': 'Data terlalu sedikit'})
        continue
    
    X = df_model[fitur_tersedia]
    y = df_model['harga_bersih']
    
    # Train-test split
    test_size = min(0.2, 10/len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross validation
    n_splits = min(5, len(X))
    if n_splits >= 2:
        cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    else:
        cv_mean = 0
        cv_std = 0
    
    results.append({
        'dataset': name,
        'r2': r2,
        'mae': mae,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'n_samples': len(df_model)
    })
    
    print(f"      R² = {r2:.4f}")
    print(f"      MAE = Rp {mae:,.0f}")
    print(f"      CV R² = {cv_mean:.4f} (+-{cv_std:.4f})")

# ============================================================
# 9. KESIMPULAN
# ============================================================
print("\n" + "="*60)
print("🎯 8. KESIMPULAN")
print("="*60)

if len(results) >= 2:
    r2_raw = results[0]['r2']
    r2_clean = results[1]['r2']
    
    print(f"\n   R² Raw   : {r2_raw:.4f}")
    print(f"   R² Clean : {r2_clean:.4f}")
    print(f"   Selisih  : {r2_raw - r2_clean:.4f}")
    
    print("\n   🔍 KEMUNGKINAN PENYEBAB:")
    
    # Cek jumlah data
    if len(df_raw) > len(df_clean) * 2:
        print("   1. ✅ Data raw lebih BANYAK (2000+) vs data clean (50-)")
        print("      → Model lebih stabil dengan data banyak")
    
    # Cek variasi harga
    if 'harga_bersih' in df_raw.columns and 'harga_bersih' in df_clean.columns:
        std_raw = df_raw['harga_bersih'].std()
        std_clean = df_clean['harga_bersih'].std()
        if std_raw > std_clean * 2:
            print("   2. ✅ Data raw punya VARIASI HARGA lebih BESAR")
            print("      → Termasuk tanah (harga murah) dan rumah mewah (harga mahal)")
    
    # Cek fitur
    if 'kecamatan' not in df_clean.columns:
        print("   3. ✅ Data clean TIDAK PUNYA kolom 'kecamatan'")
        print("      → Fitur penting hilang!")
    
    if 'tanggal_clean' not in df_clean.columns:
        print("   4. ✅ Data clean TIDAK PUNYA kolom 'tanggal_clean'")
        print("      → Tidak bisa merge dengan data makro (kurs, inflasi)")
    
    # Cek missing values
    missing_clean = df_clean.isnull().sum().sum()
    missing_raw = df_raw.isnull().sum().sum()
    if missing_clean > missing_raw:
        print("   5. ✅ Data clean punya MORE MISSING VALUES")
        print(f"      → Raw: {missing_raw}, Clean: {missing_clean}")
    
    # Cek duplikat
    dup_clean = df_clean.duplicated().sum()
    dup_raw = df_raw.duplicated().sum()
    if dup_raw > dup_clean:
        print("   6. ✅ Data raw punya BANYAK DUPLIKAT")
        print(f"      → Model hafal pola duplikat → R² tinggi palsu")
    
    # Rekomendasi
    print("\n   💡 REKOMENDASI:")
    print("   1. Gunakan data RAW untuk training (tapi hati-hati overfitting)")
    print("   2. Atau gunakan data CLEAN dengan fitur tambahan:")
    print("      - Tambahkan 'harga_per_m2_tanah'")
    print("      - Tambahkan 'rasio_lb_lt'")
    print("      - Gunakan model per kecamatan")
    print("   3. Target R² yang realistis untuk data clean: 0.6-0.75")

print("\n" + "="*80)
print("✅ CHECKER SELESAI")
print("="*80)