import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GLOBAL VARIABLES (sama dengan app.py)
# ============================================================
encoder = None
model = None
feature_columns = []
all_metrics = []

# ============================================================
# 🔥 FUNGSI PREPROCESSING (HARUS SAMA DENGAN TRAINING)
# ============================================================
def preprocess_single(kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan, encoder=None):
    """Preprocessing untuk 1 properti (tanpa data lain)"""
    
    # 1. Fitur derivatif dasar
    rasio_lb_lt = luas_bangunan / luas_tanah if luas_tanah > 0 else 0
    kepadatan_kamar = (kamar_tidur / luas_bangunan) * 100 if luas_bangunan > 0 else 0
    
    # 2. Fitur kategori (binning)
    if luas_tanah <= 100:
        kategori_luas = 0
    elif luas_tanah <= 200:
        kategori_luas = 1
    elif luas_tanah <= 500:
        kategori_luas = 2
    else:
        kategori_luas = 3
    
    if kamar_tidur <= 2:
        kategori_kamar = 0
    elif kamar_tidur <= 4:
        kategori_kamar = 1
    elif kamar_tidur <= 6:
        kategori_kamar = 2
    else:
        kategori_kamar = 3
    
    # 3. Encode kecamatan
    if encoder:
        try:
            kec_encoded = encoder.transform([kecamatan])[0]
        except:
            kec_map = {'Lowokwaru':0, 'Klojen':1, 'Blimbing':2, 'Sukun':3, 'Kedungkandang':4}
            kec_encoded = kec_map.get(kecamatan, 0)
    else:
        kec_map = {'Lowokwaru':0, 'Klojen':1, 'Blimbing':2, 'Sukun':3, 'Kedungkandang':4}
        kec_encoded = kec_map.get(kecamatan, 0)
    
    # 4. Fitur rasio (default values karena input tunggal)
    harga_ratio_kec = 1.0
    harga_per_m2_ratio = 1.0
    luas_ratio_kec = 1.0
    harga_median_kec = 0
    luas_mean_kec = luas_tanah
    harga_per_m2_median_kec = 0
    
    return {
        'kamar_tidur': kamar_tidur,
        'kamar_mandi': kamar_mandi,
        'luas_tanah_m2': luas_tanah,
        'luas_bangunan': luas_bangunan,
        'rasio_lb_lt': rasio_lb_lt,
        'kepadatan_kamar': kepadatan_kamar,
        'kategori_luas': kategori_luas,
        'kategori_kamar': kategori_kamar,
        'kecamatan_encoded': kec_encoded,
        'harga_ratio_kec': harga_ratio_kec,
        'harga_per_m2_ratio': harga_per_m2_ratio,
        'luas_ratio_kec': luas_ratio_kec,
        'harga_median_kec': harga_median_kec,
        'luas_mean_kec': luas_mean_kec,
        'harga_per_m2_median_kec': harga_per_m2_median_kec
    }


def load_model():
    """Load model optimasi untuk web"""
    global encoder, model, feature_columns, all_metrics
    
    print("="*60)
    print("🚀 LOADING MODEL OPTIMASI...")
    print("="*60)
    
    # ============================================================
    # 1. Load encoder
    # ============================================================
    encoder_path = 'models/encoder_optimasi.pkl'
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        print(f"✅ Encoder loaded from {encoder_path}")
    else:
        print(f"❌ Encoder not found at {encoder_path}")
        encoder = None
    
    # ============================================================
    # 2. Load model terbaik
    # ============================================================
    model_path = 'models/model_terbaik_optimasi.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {model_path}")
        
        # Simpan feature names dari model
        try:
            feature_columns = list(model.feature_names_in_)
            print(f"📋 Model expects {len(feature_columns)} features")
        except:
            feature_columns = []
            print(f"⚠️ Could not extract feature names from model")
    else:
        print(f"❌ Model not found at {model_path}")
        model = None
    
    # ============================================================
    # 3. Load metrics
    # ============================================================
    results_path = 'models/results_optimasi.pkl'
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            all_metrics = pickle.load(f)
        print(f"✅ Metrics loaded from {results_path}")
        print(f"📊 Total models: {len(all_metrics)}")
    else:
        print(f"❌ Metrics not found at {results_path}")
        all_metrics = []
    
    print("="*60)
    
    return model, encoder, feature_columns, all_metrics


def predict_price(kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan):
    """Prediksi harga properti dengan model terbaik"""
    global model, feature_columns, encoder
    
    if model is None:
        print("⚠️ Model not loaded! Call load_model() first.")
        return None
    
    # Preprocessing input
    input_features = preprocess_single(kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan, encoder)
    
    # Buat DataFrame dengan kolom sesuai urutan model
    if feature_columns:
        X_pred = pd.DataFrame([{col: input_features.get(col, 0) for col in feature_columns}])
    else:
        # Fallback: urutan default 15 fitur
        X_pred = pd.DataFrame([{
            'kamar_tidur': input_features['kamar_tidur'],
            'kamar_mandi': input_features['kamar_mandi'],
            'luas_tanah_m2': input_features['luas_tanah_m2'],
            'luas_bangunan': input_features['luas_bangunan'],
            'rasio_lb_lt': input_features['rasio_lb_lt'],
            'kepadatan_kamar': input_features['kepadatan_kamar'],
            'kategori_luas': input_features['kategori_luas'],
            'kategori_kamar': input_features['kategori_kamar'],
            'kecamatan_encoded': input_features['kecamatan_encoded'],
            'harga_ratio_kec': input_features['harga_ratio_kec'],
            'harga_per_m2_ratio': input_features['harga_per_m2_ratio'],
            'luas_ratio_kec': input_features['luas_ratio_kec'],
            'harga_median_kec': input_features['harga_median_kec'],
            'luas_mean_kec': input_features['luas_mean_kec'],
            'harga_per_m2_median_kec': input_features['harga_per_m2_median_kec']
        }])
    
    try:
        # Prediksi dengan log harga, lalu konversi balik
        y_pred_log = model.predict(X_pred)[0]
        harga = int(np.exp(y_pred_log))
        return harga
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def predict_price_with_model(selected_model, kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan, encoder_param=None):
    """Prediksi dengan model tertentu (untuk perbandingan)"""
    if selected_model is None:
        return None
    
    # Preprocessing input
    input_features = preprocess_single(kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan, encoder_param or encoder)
    
    # Buat DataFrame
    try:
        expected_features = list(selected_model.feature_names_in_)
        X_pred = pd.DataFrame([{col: input_features.get(col, 0) for col in expected_features}])
    except:
        X_pred = pd.DataFrame([input_features])
    
    try:
        y_pred_log = selected_model.predict(X_pred)[0]
        harga = int(np.exp(y_pred_log))
        return harga
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def get_model_metrics():
    """Mendapatkan metrics semua model"""
    global all_metrics
    return all_metrics


def get_feature_columns():
    """Mendapatkan daftar fitur yang digunakan model"""
    global feature_columns
    return feature_columns


def get_encoder():
    """Mendapatkan encoder kecamatan"""
    global encoder
    return encoder


# ============================================================
# MAIN (untuk testing)
# ============================================================
if __name__ == '__main__':
    # Load model
    load_model()
    
    if model:
        print("\n" + "="*60)
        print("🔮 TEST PREDIKSI")
        print("="*60)
        
        # Contoh prediksi
        test_cases = [
            {'kamar_tidur': 4, 'kamar_mandi': 3, 'luas_tanah': 200, 'luas_bangunan': 180, 'kecamatan': 'Blimbing'},
            {'kamar_tidur': 3, 'kamar_mandi': 2, 'luas_tanah': 120, 'luas_bangunan': 100, 'kecamatan': 'Lowokwaru'},
            {'kamar_tidur': 5, 'kamar_mandi': 4, 'luas_tanah': 300, 'luas_bangunan': 250, 'kecamatan': 'Sukun'},
        ]
        
        for test in test_cases:
            harga = predict_price(
                test['kamar_tidur'],
                test['kamar_mandi'],
                test['luas_tanah'],
                test['luas_bangunan'],
                test['kecamatan']
            )
            print(f"\n🏠 {test['kecamatan']}, KT={test['kamar_tidur']}, LT={test['luas_tanah']}m²")
            print(f"   💰 Estimasi harga: Rp {harga:,.0f}".replace(',', '.'))
        
        print("\n" + "="*60)
        print("✅ Model loader siap digunakan!")
        print("="*60)
    else:
        print("\n❌ Model gagal dimuat. Pastikan file di folder 'models/' sudah benar.")