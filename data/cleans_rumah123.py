import pandas as pd
import re
from datetime import datetime, timedelta

print("="*70)
print(" CLEANING FINAL RUMAH123 (FIXED)")
print("="*70)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('rumah123 (test4).csv')
print(f"\n[1] Data awal: {len(df)}")

# ============================================================
# 2. AMBIL KOLOM PENTING
# ============================================================
df_clean = pd.DataFrame()
df_clean['lokasi'] = df.get('text-left')
df_clean['judul'] = df.get('text-greyText')

# ============================================================
# 3. PARSE HARGA
# ============================================================
def extract_harga(text):
    try:
        text = str(text).lower().replace(',', '.').strip()
        match = re.search(r'(\d+(\.\d+)?)', text)
        if not match:
            return None
        angka = float(match.group(1))
        if 'miliar' in text or ' m' in text:
            return int(angka * 1_000_000_000)
        elif 'juta' in text or 'jt' in text:
            return int(angka * 1_000_000)
        elif 'ribu' in text:
            return int(angka * 1_000)
        else:
            angka_full = re.sub(r'[^0-9]', '', text)
            return int(angka_full) if angka_full else None
    except:
        return None

df_clean['harga_bersih'] = df.get('text-primary').apply(extract_harga)

# ============================================================
# 4. PARSE FLEX
# ============================================================
def extract_angka(text):
    try:
        match = re.search(r'\d+', str(text))
        return int(match.group()) if match else None
    except:
        return None

df_clean['kamar_tidur'] = df.get('flex').apply(extract_angka)
df_clean['kamar_mandi'] = df.get('flex 2').apply(extract_angka)
df_clean['luas_tanah_m2'] = df.get('flex 3').apply(extract_angka)
df_clean['luas_bangunan'] = df.get('flex 4').apply(extract_angka)

# ============================================================
# 5. PARSE TANGGAL
# ============================================================
today_fixed = datetime(2026, 4, 16)

bulan_map = {
    'JAN':1,'FEB':2,'MAR':3,'APR':4,'MEI':5,'JUN':6,
    'JUL':7,'AGT':8,'SEP':9,'OKT':10,'NOV':11,'DES':12
}

def parse_tanggal(t):
    try:
        t = str(t).upper().strip()
        if t == 'HARI INI':
            return today_fixed.date()
        if t == 'KEMARIN':
            return (today_fixed - timedelta(days=1)).date()
        
        match_hari = re.search(r'(\d+)\s*HARI', t)
        if match_hari:
            return (today_fixed - timedelta(days=int(match_hari.group(1)))).date()
        
        match_bulan = re.search(r'(\d+)\s*BULAN', t)
        if match_bulan:
            return (today_fixed - timedelta(days=int(match_bulan.group(1)) * 30)).date()
        
        match_tahun = re.search(r'(\d+)\s*TAHUN', t)
        if match_tahun:
            return (today_fixed - timedelta(days=int(match_tahun.group(1)) * 365)).date()
        
        match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', t)
        if match:
            return datetime(int(match.group(3)), int(match.group(2)), int(match.group(1))).date()
        
        match2 = re.search(r'(\d{1,2})\s*([A-Z]{3})', t)
        if match2:
            hari = int(match2.group(1))
            bulan = bulan_map.get(match2.group(2), 1)
            try:
                return datetime(2026, bulan, hari).date()
            except:
                return None
        return None
    except:
        return None

df_clean['tanggal_clean'] = df.get('text-3xs').apply(parse_tanggal)

# ============================================================
# 6. HAPUS DATA KOSONG
# ============================================================
df_clean = df_clean.dropna()

# ============================================================
# 7. VALIDASI DASAR
# ============================================================
df_clean = df_clean[
    (df_clean['kamar_tidur'] > 0) &
    (df_clean['kamar_mandi'] > 0) &
    (df_clean['luas_tanah_m2'] > 0) &
    (df_clean['luas_bangunan'] > 0)
]
print(f"    Setelah validasi dasar: {len(df_clean)}")

# ============================================================
# 8. FILTER NON-HUNIAN (DIPERBAIKI)
# ============================================================
# ⚠️ Hati-hati dengan pattern pendek!
non_hunian_pattern = [
    'ruko', 'gudang', 'kantor', 'pabrik', 'apartemen', 'villa',
    'tanah dijual', 'tanah murah', 'tanah luas',  # lebih spesifik
    'kavling', 'ruang usaha', 'cafe', 'resto', 'hotel',
    'dijual tanah', 'jual tanah'
]

# Filter aman: cek full phrase dulu
for pattern in non_hunian_pattern:
    mask = df_clean['judul'].str.lower().str.contains(pattern, na=False)
    df_clean = df_clean[~mask]

# Filter kos/kost TAPI jangan false match 'kosong'
mask_kos = df_clean['judul'].str.lower().str.contains(r'\bkos\b|\bkost\b', na=False, regex=True)
df_clean = df_clean[~mask_kos]

print(f"    Setelah filter non-hunian: {len(df_clean)}")

# ============================================================
# 9. MAP KECAMATAN (DIPERBAIKI)
# ============================================================
def map_kecamatan(text):
    text = str(text).upper()
    
    # Lowokwaru
    if any(x in text for x in ['LOWOKWARU', 'TLOGOMAS', 'MERJOSARI', 'SUMBERSARI',
                                 'TUNJUNGSEKAR', 'TASIKMADU', 'TUNGGULWULUNG',
                                 'MOJOLANGU', 'DINOYO', 'KETAWANGGEDE','JATIMULYO', 'TULUSREJO']):
        return 'Lowokwaru'
    
    # Klojen
    elif any(x in text for x in ['KLOJEN', 'IJEN', 'KAUMAN', 'ORO-ORO DOWO',
                                   'BARENG', 'RAMPAL CELAKET', 'KASIN', 'KIDUL DALEM', 'SAMAAN', 'KIDULDALEM', 'SUKOHARJO' , 'GADING KASRI', 'PENANGGUNGAN']):
        return 'Klojen'
    
    # Blimbing
    elif any(x in text for x in ['BLIMBING', 'ARAYA', 'SULFAT',
                                   'PANDANWANGI', 'BUNULREJO', 'POLOWIJEN', 'BALEARJOSARI', 'ARJOSARI', 'PURWODADI', 'PANDANWANGI', 'PURWANTORO', 'KESATRIAN', 'POLEHAN', 'JODIPAN']):
        return 'Blimbing'
    
    # Sukun
    elif any(x in text for x in ['SUKUN', 'DIENG', 'GADANG', 'BANDULAN',
                                   'MULYOREJO', 'PISANGCANDI', 'TIDAR', 'CIPTOMULYO', 'BANDUNGREJOSARI', 'TANJUNGREJO', 'KARANGBESUKI', 'BAKALAN KRAJAN', 'KEBONSARI']):
        return 'Sukun'
    
    # Kedungkandang
    elif any(x in text for x in ['KEDUNGKANDANG', 'MADYOPURO', 'BURING',
                                   'CEMOROKANDANG', 'WONOKOYO', 'ARJOWINANGUN', 'SAWOJAJAR', 'KOTALAMA' , 'MERGOSONO', 'BUMIAYU','KETAWANGGEDE', 'LESANPURO', 'TLOGOWARU']):
        return 'Kedungkandang'
    
    else:
        return None

df_clean['kecamatan'] = df_clean['lokasi'].apply(map_kecamatan)
df_clean = df_clean[df_clean['kecamatan'].notna()]
print(f"    Setelah map kecamatan: {len(df_clean)}")

# ============================================================
# 10. FILTER HARGA (DIPERKETAT)
# ============================================================
df_clean = df_clean[
    (df_clean['harga_bersih'] >= 300_000_000) &    # minimal 300jt
    (df_clean['harga_bersih'] <= 15_000_000_000)    # maksimal 15M
]
print(f"    Setelah filter harga: {len(df_clean)}")

# ============================================================
# 11. HAPUS ANOMALI
# ============================================================
# Rasio kamar vs luas tanah (max 10 kamar per 100m2)
df_clean = df_clean[
    (df_clean['kamar_tidur'] / df_clean['luas_tanah_m2'] * 100) <= 10
]

# Rasio luas bangunan vs tanah (max 5x)
df_clean = df_clean[
    df_clean['luas_bangunan'] <= df_clean['luas_tanah_m2'] * 5
]

# Harga per m2 masuk akal (1jt - 50jt/m2)
harga_m2 = df_clean['harga_bersih'] / df_clean['luas_tanah_m2']
df_clean = df_clean[
    (harga_m2 >= 1_000_000) & (harga_m2 <= 50_000_000)
]

print(f"    Setelah hapus anomali: {len(df_clean)}")

# ============================================================
# 12. HAPUS DUPLIKAT
# ============================================================
df_clean = df_clean.drop_duplicates(subset=['judul'])
print(f"    Setelah hapus duplikat: {len(df_clean)}")

# ============================================================
# 13. SORT & SAVE
# ============================================================
df_clean = df_clean.sort_values('tanggal_clean', ascending=False)
df_clean.to_csv('rumah123_cleans8_final.csv', index=False)

print("\n" + "="*70)
print(" CLEANING FINAL BERHASIL")
print("="*70)

print(f"\n📊 Ringkasan:")
print(f"   Total data akhir: {len(df_clean)}")
print(f"   Range harga: Rp {df_clean['harga_bersih'].min():,.0f} - Rp {df_clean['harga_bersih'].max():,.0f}")
print(f"   Range tanggal: {df_clean['tanggal_clean'].min()} - {df_clean['tanggal_clean'].max()}")
print(f"\n📊 Per Kecamatan:")
print(df_clean['kecamatan'].value_counts().to_string())