# run.py
from app import app

if __name__ == '__main__':
    print("="*60)
    print("🚀 MENJALANKAN APLIKASI PREDIKSI HARGA PROPERTI")
    print("="*60)
    print("📁 Folder data: ./data/")
    print("📁 Folder models: ./models/")
    print("🌐 Akses di: http://localhost:5000")
    print("="*60)
    print("🔐 Login Demo:")
    print("   Email: admin@example.com")
    print("   Password: admin123")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)