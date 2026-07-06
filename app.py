from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, abort
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy         
from flask_migrate import Migrate                
from flask_bcrypt import Bcrypt                  
from functools import wraps                      
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import json 
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def fromjson_filter(value):
    return json.loads(value)
app.jinja_env.filters['fromjson'] = fromjson_filter

# ============================================================
# KONFIGURASI
# ============================================================
app.secret_key = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///skripsi.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@app.context_processor
def inject_globals():
    return dict(timedelta=timedelta)

# ============================================================
# LAZY LOADING - MODEL, ENCODER, DAN DATA
# ============================================================
"""
🔑 Kunci Hemat RAM:
- Model hanya diload SAAT dipanggil (bukan di startup)
- Data CSV hanya diload SAAT dibutuhkan
- Cache disimpan setelah pertama kali dipanggil
"""

_models_cache = {}
_encoder = None
_df_properti = None
_metrics_cache = None
_feature_columns_cache = None

def load_encoder():
    """Load encoder secara lazy (hanya sekali)"""
    global _encoder
    if _encoder is None:
        encoder_path = 'models/encoder_kecamatan.pkl'
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                _encoder = pickle.load(f)
            print(f"✅ Encoder loaded")
        else:
            print(f"⚠️ Encoder not found at {encoder_path}")
    return _encoder

def load_data():
    """Load data properti secara lazy (hanya sekali)"""
    global _df_properti
    if _df_properti is None:
        data_paths = [
            'rumah123_cleans8_final.csv', 
            'data/rumah123_cleans8_final.csv',
            '../data/rumah123_cleans8_final.csv'
        ]
        for path in data_paths:
            if os.path.exists(path):
                _df_properti = pd.read_csv(path)
                print(f"✅ Data loaded: {len(_df_properti)} baris dari {path}")
                break
        if _df_properti is None:
            print("⚠️ Data properti tidak ditemukan!")
            _df_properti = pd.DataFrame()
    return _df_properti

def load_model(index):
    """Load model tertentu secara lazy"""
    model_key = f'model_{index}'
    if model_key not in _models_cache:
        joblib_path = f'models/model_{index}.joblib'
        pkl_path = f'models/model_{index}.pkl'
        
        if os.path.exists(joblib_path):
            _models_cache[model_key] = joblib.load(joblib_path)
            print(f"✅ Model {index} loaded from {joblib_path}")
        elif os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                _models_cache[model_key] = pickle.load(f)
            print(f"✅ Model {index} loaded from {pkl_path}")
        else:
            print(f"⚠️ Model {index} not found")
            _models_cache[model_key] = None
    return _models_cache[model_key]

def load_best_model():
    """Load best model secara lazy"""
    if 'best_model' not in _models_cache:
        if os.path.exists('models/model_terbaik.joblib'):
            _models_cache['best_model'] = joblib.load('models/model_terbaik.joblib')
            print(f"✅ Best model loaded")
        elif os.path.exists('models/model_terbaik.pkl'):
            with open('models/model_terbaik.pkl', 'rb') as f:
                _models_cache['best_model'] = pickle.load(f)
            print(f"✅ Best model loaded from PKL")
        else:
            print(f"⚠️ Best model not found")
            _models_cache['best_model'] = None
    return _models_cache['best_model']

def load_metrics():
    """Load metrics secara lazy"""
    global _metrics_cache
    if _metrics_cache is None:
        results_path = 'models/results_optimasi.pkl'
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                _metrics_cache = pickle.load(f)
            print(f"✅ Metrics loaded: {len(_metrics_cache)} models")
        else:
            print(f"⚠️ Metrics not found")
            _metrics_cache = []
    return _metrics_cache

def load_feature_columns():
    """Extract feature columns dari semua model secara lazy"""
    global _feature_columns_cache
    if _feature_columns_cache is None:
        _feature_columns_cache = []
        for i in range(4):
            mdl = load_model(i)
            if mdl is not None:
                try:
                    _feature_columns_cache.append(list(mdl.feature_names_in_))
                except Exception as e:
                    print(f"⚠️ Could not extract features from model {i}: {e}")
                    _feature_columns_cache.append(None)
            else:
                _feature_columns_cache.append(None)
    return _feature_columns_cache

# ============================================================
# FUNGSI PREPROCESSING
# ============================================================
def preprocess_for_prediction(kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan):
    encoder = load_encoder()
    df = load_data()
    
    # Encode kecamatan
    if encoder:
        try:
            kode_kecamatan = encoder.transform([kecamatan])[0]
        except:
            kec_map = {'Lowokwaru':0, 'Klojen':1, 'Blimbing':2, 'Sukun':3, 'Kedungkandang':4}
            kode_kecamatan = kec_map.get(kecamatan, 0)
    else:
        kec_map = {'Lowokwaru':0, 'Klojen':1, 'Blimbing':2, 'Sukun':3, 'Kedungkandang':4}
        kode_kecamatan = kec_map.get(kecamatan, 0)
    
    # Hitung fitur engineering
    rasio_lb_lt = luas_bangunan / luas_tanah if luas_tanah > 0 else 0
    total_kamar = kamar_tidur + kamar_mandi
    kt_x_km = kamar_tidur * kamar_mandi
    lt_x_lb = (luas_tanah * luas_bangunan) / 1000
    
    # Statistik kecamatan (dari training)
    kec_mean, kec_median, kec_std, kec_kelas = 0, 0, 0, 1
    if os.path.exists('models/statistik_kecamatan.pkl'):
        try:
            with open('models/statistik_kecamatan.pkl', 'rb') as f:
                statistik = pickle.load(f)
            kec_mean = statistik['harga_mean'].get(kecamatan, 0)
            kec_median = statistik['harga_median'].get(kecamatan, 0)
            kec_std = statistik['harga_std'].get(kecamatan, 0)
            kec_kelas = statistik['kelas'].get(kecamatan, 1)
        except:
            pass
    
    # Makro
    kurs_usd = 16850.0
    inflasi_malang = 3.5
    inflasi_nasional = 3.5
    if df is not None and len(df) > 0:
        if 'kurs_usd' in df.columns:
            kurs_usd = df['kurs_usd'].mean()
        if 'inflasi_malang' in df.columns:
            inflasi_malang = df['inflasi_malang'].mean()
        if 'inflasi_nasional' in df.columns:
            inflasi_nasional = df['inflasi_nasional'].mean()
    
    return {
        'kamar_tidur': int(kamar_tidur),
        'kamar_mandi': int(kamar_mandi),
        'luas_tanah_m2': float(luas_tanah),
        'luas_bangunan': float(luas_bangunan),
        'rasio_lb_lt': float(rasio_lb_lt),
        'total_kamar': int(total_kamar),
        'kt_x_km': int(kt_x_km),
        'lt_x_lb': float(lt_x_lb),
        'kurs_usd': float(kurs_usd),
        'inflasi_malang': float(inflasi_malang),
        'inflasi_nasional': float(inflasi_nasional),
        'kode_kecamatan': int(kode_kecamatan),
        'kec_mean': float(kec_mean),
        'kec_median': float(kec_median),
        'kec_std': float(kec_std),
        'kec_kelas': int(kec_kelas)
    }

def predict_price_with_model(selected_model, kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan, model_features=None):
    if selected_model is None:
        return None
    
    input_features = preprocess_for_prediction(kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan)
    
    try:
        if model_features:
            X_pred = pd.DataFrame([{col: input_features.get(col, 0) for col in model_features}])
        else:
            model_features_from_model = list(selected_model.feature_names_in_)
            X_pred = pd.DataFrame([{col: input_features.get(col, 0) for col in model_features_from_model}])
        y_pred_log = selected_model.predict(X_pred)[0]
        harga = int(np.exp(y_pred_log))
        return harga
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def predict_price(kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan):
    """Prediksi menggunakan model terbaik (lazy load)"""
    best_model = load_best_model()
    if best_model is not None:
        return predict_price_with_model(best_model, kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan, None)
    
    # Fallback ke model 3
    model_3 = load_model(3)
    if model_3 is not None:
        features = load_feature_columns()
        features_3 = features[3] if len(features) > 3 else None
        return predict_price_with_model(model_3, kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan, features_3)
    
    return None

def preprocess_dataframe(df):
    df = df.copy()
    if len(df) > 1 and 'kecamatan' in df.columns:
        harga_median_kec = df.groupby('kecamatan')['harga_bersih'].median()
        df['perbandingan_harga_kec'] = df['harga_bersih'] / df['kecamatan'].map(harga_median_kec)
    else:
        df['perbandingan_harga_kec'] = 1.0
    return df

def get_actual_price_comparison(kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan):
    df = load_data()
    if df is None or df.empty:
        return None, 0, None
    
    df = df.copy()
    harga_col = 'harga_bersih' if 'harga_bersih' in df.columns else 'harga'
    if harga_col not in df.columns:
        return None, 0, None
    
    if 'kecamatan' in df.columns:
        similar = df[df['kecamatan'] == kecamatan]
    else:
        similar = df
    
    if len(similar) == 0:
        return None, 0, None
    
    if 'kamar_tidur' in similar.columns:
        similar_tidur = similar[similar['kamar_tidur'] == kamar_tidur]
        if len(similar_tidur) >= 3:
            similar = similar_tidur
    
    if 'kamar_mandi' in similar.columns:
        similar_mandi = similar[similar['kamar_mandi'] == kamar_mandi]
        if len(similar_mandi) >= 3:
            similar = similar_mandi
    
    if 'luas_tanah_m2' in similar.columns:
        similar = similar[(similar['luas_tanah_m2'] >= luas_tanah * 0.7) & (similar['luas_tanah_m2'] <= luas_tanah * 1.3)]
    
    if 'luas_bangunan' in similar.columns:
        similar = similar[(similar['luas_bangunan'] >= luas_bangunan * 0.7) & (similar['luas_bangunan'] <= luas_bangunan * 1.3)]
    
    if len(similar) > 0:
        harga_aktual_mean = float(similar[harga_col].mean())
        jumlah_data = len(similar)
        stats = {
            'mean': harga_aktual_mean,
            'median': float(similar[harga_col].median()),
            'min': float(similar[harga_col].min()),
            'max': float(similar[harga_col].max()),
            'count': jumlah_data
        }
        return harga_aktual_mean, jumlah_data, stats
    
    return None, 0, None

# ============================================================
# MODEL DATABASE
# ============================================================
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')

    def get_id(self):
        return str(self.id)

class PredictionLog(db.Model):
    __tablename__ = 'prediction_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    input_data = db.Column(db.Text, nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)
    actual_price = db.Column(db.Float, nullable=True)

    user = db.relationship('User', backref='predictions')

# ============================================================
# LOAD USER UNTUK FLASK-LOGIN
# ============================================================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ============================================================
# DECORATOR ADMIN
# ============================================================
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        user = User.query.get(int(current_user.get_id()))
        if user.role != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# METRICS FUNCTION
# ============================================================
def calculate_metrics():
    df = load_data()
    if df is None or df.empty:
        return {
            'total_data': 0,
            'all_models': [],
            'kecamatan_counts': {},
            'price_stats': {},
            'last_update': datetime.now().strftime('%d %b %Y')
        }
    
    models_metrics = []
    all_metrics = load_metrics()
    if all_metrics:
        sorted_metrics = sorted(enumerate(all_metrics), key=lambda x: x[1].get('r2', 0), reverse=True)
        for rank, (original_id, metric) in enumerate(sorted_metrics):
            models_metrics.append({
                'id': original_id,
                'rank': rank + 1,
                'name': metric.get('name', f'Model {original_id+1}'),
                'n_features': metric.get('n_features', 0),
                'mae': metric.get('mae', 0),
                'r2': metric.get('r2', 0),
                'mape': metric.get('mape', 0) * 100 if metric.get('mape', 0) < 1 else metric.get('mape', 0),
                'cv_r2_mean': metric.get('cv_r2_mean', 0),
                'is_best': rank == 0
            })
    else:
        for i in range(4):
            models_metrics.append({
                'id': i,
                'rank': i + 1,
                'name': f'Model {i+1}',
                'n_features': 0,
                'mae': 0,
                'r2': 0,
                'mape': 0,
                'cv_r2_mean': 0,
                'is_best': i == 0
            })
    
    if 'kecamatan' in df.columns:
        kec_counts = df['kecamatan'].value_counts().to_dict()
        harga_col = 'harga_bersih' if 'harga_bersih' in df.columns else 'harga'
        if harga_col in df.columns:
            price_stats = {}
            for kec, group in df.groupby('kecamatan'):
                price_stats[str(kec)] = {
                    'mean': float(group[harga_col].mean()),
                    'min': float(group[harga_col].min()),
                    'max': float(group[harga_col].max()),
                    'count': int(len(group))
                }
        else:
            price_stats = {}
    else:
        kec_counts = {}
        price_stats = {}
    
    if 'tanggal_clean' in df.columns:
        try:
            tanggal = pd.to_datetime(df['tanggal_clean'], errors='coerce').dropna()
            date_range = {
                'from': tanggal.min().strftime('%d %b %Y'),
                'to': tanggal.max().strftime('%d %b %Y')
            }
        except:
            date_range = None
    else:
        date_range = None

    return {
        'total_data': len(df),
        'all_models': models_metrics,
        'kecamatan_counts': kec_counts,
        'price_stats': price_stats,
        'last_update': datetime.now().strftime('%d %b %Y'),
        'date_range': date_range
    }

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login berhasil!', 'success')
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('dashboard'))
        else:
            flash('Email atau password salah!', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Email sudah terdaftar!', 'danger')
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            new_user = User(email=email, name=name, password=hashed_password, role='user')
            db.session.add(new_user)
            db.session.commit()
            flash('Registrasi berhasil! Silakan login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Anda telah logout.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    metrics = calculate_metrics()
    return render_template('dashboard.html', metrics=metrics, user=current_user)

@app.route('/prediksi', methods=['GET', 'POST'])
@login_required
def prediksi():
    return redirect(url_for('prediksi_with_model', model_id=3))

@app.route('/prediksi/<int:model_id>', methods=['GET', 'POST'])
@login_required
def prediksi_with_model(model_id):
    if model_id < 0 or model_id >= 4:
        model_id = 3
    
    # Lazy load model dan metrics
    selected_model = load_model(model_id)
    feature_columns = load_feature_columns()
    all_metrics = load_metrics()
    
    selected_features = feature_columns[model_id] if model_id < len(feature_columns) else None
    selected_metric = all_metrics[model_id] if model_id < len(all_metrics) else None
    
    prediction = None
    input_data = None
    
    if request.method == 'POST':
        try:
            input_data = {
                'kamar_tidur': int(request.form.get('kamar_tidur', 3)),
                'kamar_mandi': int(request.form.get('kamar_mandi', 2)),
                'luas_tanah': float(request.form.get('luas_tanah', 100)),
                'luas_bangunan': float(request.form.get('luas_bangunan', 80)),
                'kecamatan': request.form.get('kecamatan', 'Lowokwaru')
            }
            
            if selected_model:
                harga_prediksi = predict_price_with_model(
                    selected_model,
                    input_data['kamar_tidur'],
                    input_data['kamar_mandi'],
                    input_data['luas_tanah'],
                    input_data['luas_bangunan'],
                    input_data['kecamatan'],
                    selected_features
                )
            else:
                harga_prediksi = predict_price(
                    input_data['kamar_tidur'],
                    input_data['kamar_mandi'],
                    input_data['luas_tanah'],
                    input_data['luas_bangunan'],
                    input_data['kecamatan']
                )
            
            if harga_prediksi:
                harga_aktual, jumlah_data, stats = get_actual_price_comparison(
                    input_data['kamar_tidur'],
                    input_data['kamar_mandi'],
                    input_data['luas_tanah'],
                    input_data['luas_bangunan'],
                    input_data['kecamatan']
                )
                
                model_name = selected_metric.get('name', f'Model {model_id+1}') if selected_metric else f'Model {model_id+1}'
                prediction = {
                    'harga': harga_prediksi,
                    'harga_formatted': f"Rp {harga_prediksi:,.0f}".replace(',', '.'),
                    'input': input_data,
                    'model_name': model_name,
                    'model_r2': selected_metric.get('r2', None) if selected_metric else None,
                    'model_mape': selected_metric.get('mape', None) if selected_metric else None
                }
                
                if harga_aktual:
                    selisih = abs(harga_prediksi - harga_aktual)
                    persentase_selisih = (selisih / harga_aktual) * 100
                    prediction.update({
                        'harga_aktual': harga_aktual,
                        'harga_aktual_formatted': f"Rp {harga_aktual:,.0f}".replace(',', '.'),
                        'selisih': selisih,
                        'selisih_formatted': f"Rp {selisih:,.0f}".replace(',', '.'),
                        'persentase_selisih': persentase_selisih,
                        'akurasi': max(0, 100 - persentase_selisih),
                        'jumlah_data_pembanding': jumlah_data,
                        'stats': stats
                    })
                
                # Simpan log
                try:
                    input_json = json.dumps(input_data)
                    log_entry = PredictionLog(
                        user_id=current_user.id,
                        input_data=input_json,
                        predicted_price=harga_prediksi,
                        actual_price=harga_aktual if harga_aktual else None
                    )
                    db.session.add(log_entry)
                    db.session.commit()
                except Exception as e:
                    print(f"⚠️ Gagal menyimpan log: {e}")
            else:
                flash('Gagal melakukan prediksi. Silakan coba lagi.', 'danger')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    
    metrics = calculate_metrics()
    return render_template('prediksi.html', 
                         prediction=prediction, 
                         user=current_user,
                         selected_model_id=model_id,
                         metrics=metrics)

# ============================================================
# ROUTE ADMIN
# ============================================================
@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    all_users = User.query.all()
    all_logs = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).all()
    df = load_data()
    total_data = len(df) if df is not None else 0
    return render_template('admin_dashboard.html', users=all_users, logs=all_logs, total_data=total_data)

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    all_users = User.query.all()
    return render_template('admin_users.html', users=all_users)

@app.route('/admin/logs')
@login_required
@admin_required
def admin_logs():
    all_logs = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).all()
    return render_template('admin_logs.html', logs=all_logs)

@app.route('/admin/user/delete/<int:user_id>')
@login_required
@admin_required
def admin_delete_user(user_id):
    user_to_delete = User.query.get(user_id)
    if user_to_delete and user_to_delete.email != current_user.email:
        db.session.delete(user_to_delete)
        db.session.commit()
        flash(f'User {user_to_delete.email} berhasil dihapus', 'success')
    else:
        flash('Tidak dapat menghapus user ini', 'danger')
    return redirect(url_for('admin_users'))

# ============================================================
# API ROUTES
# ============================================================
@app.route('/api/metrics')
def api_metrics():
    return jsonify(calculate_metrics())

@app.route('/api/data')
def api_data():
    df = load_data()
    if df is None or df.empty:
        return jsonify({'error': 'No data'})
    
    sample_size = min(50, len(df))
    sample = df.sample(sample_size).copy()
    sample = preprocess_dataframe(sample)
    
    encoder = load_encoder()
    if encoder and 'kecamatan' in sample.columns:
        try:
            sample['kode_kecamatan'] = encoder.transform(sample['kecamatan'])
        except:
            kec_map = {'Lowokwaru':0, 'Klojen':1, 'Blimbing':2, 'Sukun':3, 'Kedungkandang':4}
            sample['kode_kecamatan'] = sample['kecamatan'].map(kec_map).fillna(0)
    
    # Gunakan best model untuk prediksi
    pred_model = load_best_model()
    feature_columns = load_feature_columns()
    pred_features = feature_columns[3] if len(feature_columns) > 3 else None
    
    if pred_model and pred_features:
        try:
            X_sample = pd.DataFrame()
            for col in pred_features:
                if col in sample.columns:
                    X_sample[col] = sample[col]
                else:
                    X_sample[col] = 0
            y_pred_log = pred_model.predict(X_sample)
            y_pred = np.exp(y_pred_log)
        except Exception as e:
            print(f"Prediction error in API: {e}")
            y_pred = sample['harga_bersih'].values if 'harga_bersih' in sample.columns else [0] * len(sample)
    else:
        y_pred = sample['harga_bersih'].values if 'harga_bersih' in sample.columns else [0] * len(sample)
    
    return jsonify({
        'actual': sample['harga_bersih'].tolist() if 'harga_bersih' in sample.columns else [0] * len(sample),
        'predicted': y_pred.tolist(),
        'kecamatan': sample['kecamatan'].tolist() if 'kecamatan' in sample.columns else ['Unknown'] * len(sample)
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    harga = predict_price(
        data.get('kamar_tidur', 3),
        data.get('kamar_mandi', 2),
        data.get('luas_tanah', 100),
        data.get('luas_bangunan', 80),
        data.get('kecamatan', 'Lowokwaru')
    )
    return jsonify({
        'success': True,
        'harga': harga,
        'harga_formatted': f"Rp {harga:,.0f}".replace(',', '.') if harga else None
    })

@app.route('/api/get_nilai_properti', methods=['POST'])
def api_get_nilai_properti():
    df = load_data()
    
    try:
        data = request.get_json()
        kecamatan = data.get('kecamatan')
        luas_tanah = data.get('luas_tanah')
        luas_bangunan = data.get('luas_bangunan')
        kamar_tidur = data.get('kamar_tidur')
        kamar_mandi = data.get('kamar_mandi')
        
        df_filtered = df[df['kecamatan'] == kecamatan]
        
        if luas_tanah and luas_tanah != '':
            df_filtered = df_filtered[df_filtered['luas_tanah_m2'] == int(luas_tanah)]
        if luas_bangunan and luas_bangunan != '':
            df_filtered = df_filtered[df_filtered['luas_bangunan'] == int(luas_bangunan)]
        if kamar_tidur and kamar_tidur != '':
            df_filtered = df_filtered[df_filtered['kamar_tidur'] == int(kamar_tidur)]
        if kamar_mandi and kamar_mandi != '':
            df_filtered = df_filtered[df_filtered['kamar_mandi'] == int(kamar_mandi)]
        
        luas_tanah_list = sorted(df_filtered['luas_tanah_m2'].dropna().unique().tolist())
        luas_bangunan_list = sorted(df_filtered['luas_bangunan'].dropna().unique().tolist())
        kamar_tidur_list = sorted(df_filtered['kamar_tidur'].dropna().unique().tolist())
        kamar_mandi_list = sorted(df_filtered['kamar_mandi'].dropna().unique().tolist())
        
        return jsonify({
            'success': True,
            'data': {
                'luas_tanah': luas_tanah_list,
                'luas_bangunan': luas_bangunan_list,
                'kamar_tidur': kamar_tidur_list,
                'kamar_mandi': kamar_mandi_list
            },
            'count': len(df_filtered)
        })
        
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ============================================================
# MAIN - PRODUCTION READY
# ============================================================
if __name__ == '__main__':
    import os
    
    # 🔥 PERUBAHAN PENTING UNTUK DEPLOYMENT
    port = int(os.environ.get('PORT', 8000))  # ← Gunakan port 8000
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print("\n" + "="*60)
    print("🚀 APLIKASI PREDIKSI HARGA PROPERTI")
    print("="*60)
    print(f"🌐 Port: {port}")
    print(f"🔧 Debug: {debug_mode}")
    print(f"💾 RAM Mode: Lazy Loading (hemat memory)")
    print("="*60)
    
    # 🔥 Buat database jika belum ada
    with app.app_context():
        db.create_all()
        # Buat admin default jika belum ada
        if not User.query.filter_by(email='admin@example.com').first():
            hashed = bcrypt.generate_password_hash('admin123').decode('utf-8')
            admin = User(email='admin@example.com', name='Admin', password=hashed, role='admin')
            db.session.add(admin)
            db.session.commit()
            print("✅ Admin default created: admin@example.com / admin123")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)