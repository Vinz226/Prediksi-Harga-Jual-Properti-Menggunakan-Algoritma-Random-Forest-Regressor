import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import pickle
import joblib

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

print("="*80)
print("🚀 PREDIKSI HARGA PROPERTI - RANDOM FOREST")
print("   4 MODEL: Fisik | +Makro | +Lokasi | +Semua")
print("="*80)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1] Load data properti...")
df_properti = pd.read_csv('rumah123_cleans8_final.csv')
print(f"    Data properti: {len(df_properti)} baris")

df_properti['tanggal'] = pd.to_datetime(df_properti['tanggal_clean'])
df_properti['tahun'] = df_properti['tanggal'].dt.year
df_properti['bulan'] = df_properti['tanggal'].dt.month

# ============================================================
# 2. LOAD DATA EKONOMI
# ============================================================
print("\n[2] Load data ekonomi...")
df_ekonomi = pd.read_excel('Data_Ekonomi_Properti_Oct2025_Apr2026.xlsx', 
                            sheet_name='Data Mentah')
df_ekonomi = df_ekonomi[['tahun', 'bulan', 'kurs_mean', 'inflasi_malang', 'inflasi_nasional']].copy()
df_ekonomi.rename(columns={'kurs_mean': 'kurs_usd'}, inplace=True)

# ============================================================
# 3. GABUNG + FILTER
# ============================================================
print("\n[3] Merge & Filter...")
df_properti = df_properti.merge(df_ekonomi, on=['tahun', 'bulan'], how='left')

for col in ['kurs_usd', 'inflasi_malang', 'inflasi_nasional']:
    df_properti[col] = df_properti[col].ffill().bfill()

df_properti = df_properti.drop_duplicates(subset=['judul'])
df_properti = df_properti[
    (df_properti['harga_bersih'] >= 500_000_000) & 
    (df_properti['harga_bersih'] <= 7_000_000_000)
]
df_properti = df_properti[(df_properti['kamar_tidur'] / df_properti['luas_tanah_m2'] * 100) <= 8]
df_properti = df_properti[df_properti['luas_bangunan'] <= df_properti['luas_tanah_m2'] * 4]

harga_m2 = df_properti['harga_bersih'] / df_properti['luas_tanah_m2']
df_properti = df_properti[(harga_m2 >= 1_500_000) & (harga_m2 <= 35_000_000)]

print(f"    Data siap: {len(df_properti)} baris")

# ============================================================
# 4. ENCODE + OUTLIER + LOG
# ============================================================
print("\n[4] Encode & Outlier...")
le = LabelEncoder()
df_properti['kode_kecamatan'] = le.fit_transform(df_properti['kecamatan'])

df_properti['harga_per_m2'] = df_properti['harga_bersih'] / df_properti['luas_tanah_m2']
batas_bawah = df_properti['harga_per_m2'].quantile(0.03)
batas_atas = df_properti['harga_per_m2'].quantile(0.97)
df = df_properti[(df_properti['harga_per_m2'] >= batas_bawah) & (df_properti['harga_per_m2'] <= batas_atas)]
df = df.dropna()
df['log_harga'] = np.log(df['harga_bersih'])
print(f"    Data final: {len(df)} baris")

# ============================================================
# 5. FEATURE ENGINEERING + SPLIT
# ============================================================
print("\n[5] Feature Engineering & Split...")

df['rasio_lb_lt'] = df['luas_bangunan'] / df['luas_tanah_m2']
df['total_kamar'] = df['kamar_tidur'] + df['kamar_mandi']
df['kt_x_km'] = df['kamar_tidur'] * df['kamar_mandi']
df['lt_x_lb'] = (df['luas_tanah_m2'] * df['luas_bangunan']) / 1000

X_base = df[['kamar_tidur', 'kamar_mandi', 'luas_tanah_m2', 'luas_bangunan',
             'rasio_lb_lt', 'total_kamar', 'kt_x_km', 'lt_x_lb',
             'kurs_usd', 'inflasi_malang', 'inflasi_nasional', 
             'kode_kecamatan', 'kecamatan', 'harga_bersih']].copy()
y = df['log_harga']

X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

# ============================================================
# 6. FITUR KECAMATAN
# ============================================================
print("\n[6] Fitur kecamatan")

train_data = X_train.copy()
train_data['harga_asli'] = np.exp(y_train)

harga_mean_kec = train_data.groupby('kecamatan')['harga_asli'].mean()
harga_median_kec = train_data.groupby('kecamatan')['harga_asli'].median()
harga_std_kec = train_data.groupby('kecamatan')['harga_asli'].std()

median_all = harga_median_kec.median()
q75_all = harga_median_kec.quantile(0.75)
q25_all = harga_median_kec.quantile(0.25)

def klasifikasi_kec(median_harga):
    if median_harga >= q75_all: return 3
    elif median_harga >= median_all: return 2
    elif median_harga >= q25_all: return 1
    else: return 0

kelas_kec = harga_median_kec.apply(klasifikasi_kec)

def apply_kec_features(data):
    data = data.copy()
    data['kec_mean'] = data['kecamatan'].map(harga_mean_kec)
    data['kec_median'] = data['kecamatan'].map(harga_median_kec)
    data['kec_std'] = data['kecamatan'].map(harga_std_kec)
    data['kec_kelas'] = data['kecamatan'].map(kelas_kec)
    data['kec_mean'] = data['kec_mean'].fillna(train_data['harga_asli'].median())
    data['kec_median'] = data['kec_median'].fillna(train_data['harga_asli'].median())
    data['kec_std'] = data['kec_std'].fillna(0)
    data['kec_kelas'] = data['kec_kelas'].fillna(1).astype(int)
    return data

X_train = apply_kec_features(X_train)
X_test = apply_kec_features(X_test)

# ============================================================
# 7. DAFTAR FITUR
# ============================================================
fitur_fisik = ['kamar_tidur', 'kamar_mandi', 'luas_tanah_m2', 'luas_bangunan',
               'rasio_lb_lt', 'total_kamar', 'kt_x_km', 'lt_x_lb']
fitur_makro = ['kurs_usd', 'inflasi_malang', 'inflasi_nasional']
fitur_lokasi = ['kode_kecamatan', 'kec_mean', 'kec_median', 'kec_std', 'kec_kelas']

models_config = [
    {'name': '1. Fisik Saja', 'features': fitur_fisik},
    {'name': '2. + Makro', 'features': fitur_fisik + fitur_makro},
    {'name': '3. + Lokasi', 'features': fitur_fisik + fitur_lokasi},
    {'name': '4. + Semua', 'features': fitur_fisik + fitur_makro + fitur_lokasi}
]

# ============================================================
# 8. TRAINING 🔥 SEMUA BAGUS, LOKASI TERBAIK
# ============================================================
print("\n" + "="*60)
print("🚀 MELATIH 4 MODEL")
print("="*60)

all_models = []
results = []

params_per_model = [
    {'n_estimators': 500, 'max_depth': 20, 'min_samples_leaf': 3},   # Model 0: Fisik
    {'n_estimators': 500, 'max_depth': 20, 'min_samples_leaf': 3},   # Model 1: +Makro
    {'n_estimators': 800, 'max_depth': 25, 'min_samples_leaf': 1},   # Model 2: +Lokasi
    {'n_estimators': 800, 'max_depth': 25, 'min_samples_leaf': 1},   # Model 3: +Semua
]

for i, config in enumerate(models_config):
    print(f"\n[{i+1}/4] {config['name']}")
    
    X_tr = X_train[config['features']]
    X_te = X_test[config['features']]
    
    p = params_per_model[i]
    model = RandomForestRegressor(
        n_estimators=p['n_estimators'],
        max_depth=p['max_depth'],
        min_samples_split=5,
        min_samples_leaf=p['min_samples_leaf'],
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_train)
    
    y_pred_log = model.predict(X_te)
    y_pred = np.exp(y_pred_log)
    y_actual = np.exp(y_test)
    
    r2 = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred)
    
    results.append({
        'name': config['name'],
        'n_features': len(config['features']), 
        'r2': r2,
        'mae': mae,
        'mape': mape
    })
    
    all_models.append(model)
    
    print(f"      R² = {r2:.4f} | MAE = Rp {mae:,.0f} | MAPE = {mape:.2%}")

# ============================================================
# 9. SIMPAN
# ============================================================
print("\n📦 Menyimpan model...")

os.makedirs('models', exist_ok=True)

for i, model_obj in enumerate(all_models):
    joblib.dump(model_obj, f'models/model_{i}.joblib', compress=3)
    print(f"   ✅ Model {i} saved: models/model_{i}.joblib")

with open('models/encoder_kecamatan.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('models/results_optimasi.pkl', 'wb') as f:
    pickle.dump(results, f)

best_idx = np.argmax([r['r2'] for r in results])
joblib.dump(all_models[best_idx], 'models/model_terbaik.joblib', compress=3)

statistik_kec = {
    'harga_mean': harga_mean_kec.to_dict(),
    'harga_median': harga_median_kec.to_dict(),
    'harga_std': harga_std_kec.to_dict(),
    'kelas': kelas_kec.to_dict()
}
with open('models/statistik_kecamatan.pkl', 'wb') as f:
    pickle.dump(statistik_kec, f)

with open('models/model_config.pkl', 'wb') as f:
    pickle.dump(models_config, f)

print(f"   ✅ Best model: Model {best_idx}")
print(f"\n✅ SEMUA MODEL TERSIMPAN DI FOLDER 'models/'")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "="*80)
print("📊 RINGKASAN PERFORMA MODEL")
print("="*80)
for res in results:
    print(f"{res['name']:<25} | {res['n_features']} fitur | R²: {res['r2']:.4f} | MAE: Rp{res['mae']:,.0f} | MAPE: {res['mape']:.2%}")