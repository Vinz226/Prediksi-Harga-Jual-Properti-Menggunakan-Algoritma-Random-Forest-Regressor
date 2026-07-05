import pandas as pd
import numpy as np

print("="*80)
print("📊 MEMBUAT FILE EXCEL DATA EKONOMI BULANAN")
print("   Range: Oktober 2025 - April 2026")
print("="*80)

# ============================================================
# 1. LOAD DAN PROSES SEMUA DATA EKONOMI
# ============================================================

bulan_map = {
    'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
    'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
    'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
}

bulan_list = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
              'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']

# ----------------------------------------------------------
# 1a. Inflasi Nasional (Data Inflasi.xlsx)
# ----------------------------------------------------------
print("\n[1] Load Inflasi Nasional...")
df_inflasi_nasional = pd.read_excel('Data Inflasi.xlsx', sheet_name='Data Inflasi', header=4)
df_inflasi_nasional = df_inflasi_nasional.dropna(subset=['Periode', 'Data Inflasi'])

def parse_periode(periode_str):
    parts = str(periode_str).strip().split()
    return pd.Series([int(parts[1]), bulan_map[parts[0]]])

df_inflasi_nasional[['tahun', 'bulan']] = df_inflasi_nasional['Periode'].apply(parse_periode)
df_inflasi_nasional['inflasi_nasional'] = df_inflasi_nasional['Data Inflasi'].astype(str).str.replace('%', '').str.strip().astype(float)
df_inflasi_nasional = df_inflasi_nasional[['tahun', 'bulan', 'inflasi_nasional']].sort_values(['tahun', 'bulan'])
print(f"   Data tersedia: {len(df_inflasi_nasional)} bulan")
print(df_inflasi_nasional.to_string())

# ----------------------------------------------------------
# 1b. Inflasi Malang 2025
# ----------------------------------------------------------
print("\n[2] Load Inflasi Malang 2025...")
df_malang_2025 = pd.read_csv('Inflasi Year on Year (YoY) Kota Malang, 2025.csv', skiprows=3, header=None)
df_malang_2025.columns = ['Bulan', 'Inflasi']
df_malang_2025['tahun'] = 2025
df_malang_2025['bulan'] = df_malang_2025['Bulan'].map(bulan_map)
df_malang_2025['inflasi_malang'] = pd.to_numeric(df_malang_2025['Inflasi'], errors='coerce')
df_malang_2025 = df_malang_2025.dropna(subset=['inflasi_malang'])
df_malang_2025 = df_malang_2025[['tahun', 'bulan', 'inflasi_malang']]
print(f"   Data tersedia: {len(df_malang_2025)} bulan")

# ----------------------------------------------------------
# 1c. Inflasi Malang 2026
# ----------------------------------------------------------
print("\n[3] Load Inflasi Malang 2026...")
df_malang_2026 = pd.read_csv('Inflasi Year on Year (YoY) Kota Malang, 2026.csv', skiprows=3, header=None)
df_malang_2026.columns = ['Bulan', 'Inflasi']
df_malang_2026['tahun'] = 2026
df_malang_2026['bulan'] = df_malang_2026['Bulan'].map(bulan_map)
df_malang_2026['inflasi_malang'] = pd.to_numeric(df_malang_2026['Inflasi'], errors='coerce')
df_malang_2026 = df_malang_2026.dropna(subset=['inflasi_malang'])
df_malang_2026 = df_malang_2026[['tahun', 'bulan', 'inflasi_malang']]
print(f"   Data tersedia: {len(df_malang_2026)} bulan")

# Gabung inflasi Malang
df_inflasi_malang = pd.concat([df_malang_2025, df_malang_2026], ignore_index=True)
df_inflasi_malang = df_inflasi_malang.sort_values(['tahun', 'bulan'])

# ----------------------------------------------------------
# 1d. Kurs JISDOR
# ----------------------------------------------------------
print("\n[4] Load Kurs JISDOR...")
df_kurs = pd.read_excel('Informasi Kurs Jisdor.xlsx', sheet_name='Informasi Kurs Jisdor', header=4)
df_kurs = df_kurs.dropna(subset=['Tanggal', 'Kurs'])
df_kurs['Tanggal'] = pd.to_datetime(df_kurs['Tanggal'])
df_kurs['tahun'] = df_kurs['Tanggal'].dt.year
df_kurs['bulan'] = df_kurs['Tanggal'].dt.month

# Hitung rata-rata kurs per bulan
df_kurs_bulanan = df_kurs.groupby(['tahun', 'bulan'])['Kurs'].agg(['mean', 'min', 'max']).reset_index()
df_kurs_bulanan.columns = ['tahun', 'bulan', 'kurs_mean', 'kurs_min', 'kurs_max']
df_kurs_bulanan = df_kurs_bulanan.sort_values(['tahun', 'bulan'])
print(f"   Data harian: {len(df_kurs)} hari")
print(f"   Data bulanan: {len(df_kurs_bulanan)} bulan")

# ============================================================
# 2. GABUNG SEMUA DATA EKONOMI
# ============================================================
print("\n[5] Menggabungkan semua data ekonomi...")
df_ekonomi = df_kurs_bulanan.merge(df_inflasi_malang, on=['tahun', 'bulan'], how='left')
df_ekonomi = df_ekonomi.merge(df_inflasi_nasional, on=['tahun', 'bulan'], how='left')
df_ekonomi = df_ekonomi.sort_values(['tahun', 'bulan']).reset_index(drop=True)

# Interpolasi untuk mengisi missing values
df_ekonomi['inflasi_malang'] = df_ekonomi['inflasi_malang'].interpolate(method='linear', limit_direction='both')
df_ekonomi['inflasi_nasional'] = df_ekonomi['inflasi_nasional'].interpolate(method='linear', limit_direction='both')

print(f"   Data ekonomi gabungan: {len(df_ekonomi)} bulan")
print(f"   Range: {df_ekonomi['tahun'].min()}-{df_ekonomi['bulan'].min():02d} s/d {df_ekonomi['tahun'].max()}-{df_ekonomi['bulan'].max():02d}")

# Sebelum merge, cek data inflasi
print("\n🔍 CEK DATA INFLASI SEBELUM MERGE:")
print("Inflasi Malang:")
print(df_inflasi_malang.to_string())
print("\nInflasi Nasional:")
print(df_inflasi_nasional.to_string())

# Setelah merge, sebelum apapun
print("\n🔍 SETELAH MERGE (SEBELUM INTERPOLASI):")
# Cek nilai NaN
print(f"Jumlah NaN inflasi_malang: {df_ekonomi['inflasi_malang'].isnull().sum()}")
print(f"Jumlah NaN inflasi_nasional: {df_ekonomi['inflasi_nasional'].isnull().sum()}")

# Tampilkan baris yang NaN
print("\nBaris dengan NaN:")
print(df_ekonomi[df_ekonomi['inflasi_malang'].isnull() | df_ekonomi['inflasi_nasional'].isnull()])

# ============================================================
# 3. FILTER UNTUK RANGE DATA PROPERTI (Oktober 2025 - April 2026)
# ============================================================
print("\n[6] Filter untuk range properti: Oktober 2025 - April 2026...")

# Buat daftar bulan yang dibutuhkan
bulan_properti = [
    (2025, 10), (2025, 11), (2025, 12),
    (2026, 1), (2026, 2), (2026, 3), (2026, 4)
]

# Filter
df_final = df_ekonomi[
    ((df_ekonomi['tahun'] == 2025) & (df_ekonomi['bulan'] >= 10)) |
    ((df_ekonomi['tahun'] == 2026) & (df_ekonomi['bulan'] <= 4))
].copy()

# Cek apakah semua bulan properti ada
for tahun, bulan in bulan_properti:
    ada = ((df_final['tahun'] == tahun) & (df_final['bulan'] == bulan)).any()
    if not ada:
        # Cari data terdekat
        print(f"   ⚠️ {tahun}-{bulan:02d} tidak ada, menggunakan data terdekat...")
        # Cari baris terdekat dari df_ekonomi
        df_ekonomi['jarak'] = abs((df_ekonomi['tahun']*12 + df_ekonomi['bulan']) - (tahun*12 + bulan))
        nearest = df_ekonomi.loc[df_ekonomi['jarak'].idxmin()]
        new_row = {
            'tahun': tahun,
            'bulan': bulan,
            'kurs_mean': nearest['kurs_mean'],
            'kurs_min': nearest['kurs_min'],
            'kurs_max': nearest['kurs_max'],
            'inflasi_malang': nearest['inflasi_malang'],
            'inflasi_nasional': nearest['inflasi_nasional']
        }
        df_final = pd.concat([df_final, pd.DataFrame([new_row])], ignore_index=True)
        df_ekonomi = df_ekonomi.drop(columns=['jarak'])

df_final = df_final.sort_values(['tahun', 'bulan']).reset_index(drop=True)

# Tambah kolom periode label
df_final['Periode'] = df_final.apply(lambda x: f"{bulan_list[int(x['bulan'])-1]} {int(x['tahun'])}", axis=1)

# Bulatkan angka untuk tampilan rapi
df_final['kurs_mean'] = df_final['kurs_mean'].round(0).astype(int)
df_final['kurs_min'] = df_final['kurs_min'].round(0).astype(int)
df_final['kurs_max'] = df_final['kurs_max'].round(0).astype(int)
df_final['inflasi_malang'] = df_final['inflasi_malang'].round(2)
df_final['inflasi_nasional'] = df_final['inflasi_nasional'].round(2)

# ============================================================
# 4. BUAT SHEET 1: DATA EKONOMI BULANAN (RINGKAS)
# ============================================================
print("\n[7] Membuat Sheet 1: Data Ekonomi Bulanan...")

df_sheet1 = df_final[['Periode', 'tahun', 'bulan', 'kurs_mean', 'kurs_min', 'kurs_max', 
                       'inflasi_malang', 'inflasi_nasional']].copy()
df_sheet1.columns = ['Periode', 'Tahun', 'Bulan', 'Kurs Rata-rata (Rp)', 'Kurs Min (Rp)', 
                      'Kurs Max (Rp)', 'Inflasi Malang (%)', 'Inflasi Nasional (%)']

# ============================================================
# 5. BUAT SHEET 2: DATA KURS HARIAN
# ============================================================
print("\n[8] Membuat Sheet 2: Data Kurs Harian...")

# Filter kurs harian untuk range properti
df_kurs_harian = df_kurs[
    ((df_kurs['tahun'] == 2025) & (df_kurs['bulan'] >= 10)) |
    ((df_kurs['tahun'] == 2026) & (df_kurs['bulan'] <= 4))
].copy()

df_kurs_harian = df_kurs_harian[['Tanggal', 'Kurs']].sort_values('Tanggal')
df_kurs_harian.columns = ['Tanggal', 'Kurs (Rp)']

# ============================================================
# 6. BUAT SHEET 3: DATA INFLASI DETAIL
# ============================================================
print("\n[9] Membuat Sheet 3: Data Inflasi Detail...")

df_inflasi_detail = df_final[['Periode', 'tahun', 'bulan', 'inflasi_malang', 'inflasi_nasional']].copy()
df_inflasi_detail.columns = ['Periode', 'Tahun', 'Bulan', 'Inflasi Malang (%)', 'Inflasi Nasional (%)']

# Tambah keterangan sumber data
df_inflasi_detail['Sumber Inflasi Malang'] = df_inflasi_detail.apply(
    lambda x: 'Data Real' if ((x['Tahun']==2025) or (x['Tahun']==2026 and x['Bulan']<=3)) else 'Interpolasi', axis=1
)
df_inflasi_detail['Sumber Inflasi Nasional'] = df_inflasi_detail.apply(
    lambda x: 'Data Real' if ((x['Tahun']==2025 and x['Bulan']>=8) or (x['Tahun']==2026 and x['Bulan']<=3)) else 'Interpolasi', axis=1
)

# ============================================================
# 7. SIMPAN KE EXCEL
# ============================================================
print("\n[10] Menyimpan ke Excel...")

output_file = 'Data_Ekonomi_Properti_Oct2025_Apr2026.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Sheet 1: Ringkasan Bulanan
    df_sheet1.to_excel(writer, sheet_name='Ringkasan Bulanan', index=False)
    
    # Sheet 2: Kurs Harian
    df_kurs_harian.to_excel(writer, sheet_name='Kurs Harian', index=False)
    
    # Sheet 3: Inflasi Detail
    df_inflasi_detail.to_excel(writer, sheet_name='Inflasi Detail', index=False)
    
    # Sheet 4: Data Gabungan Mentah
    df_final.to_excel(writer, sheet_name='Data Mentah', index=False)

print(f"\n✅ File berhasil disimpan: {output_file}")
print(f"   Sheet 1: Ringkasan Bulanan (7 bulan)")
print(f"   Sheet 2: Kurs Harian ({len(df_kurs_harian)} hari)")
print(f"   Sheet 3: Inflasi Detail (7 bulan)")
print(f"   Sheet 4: Data Mentah")

# ============================================================
# 8. TAMPILKAN RINGKASAN
# ============================================================
print("\n" + "="*80)
print("📊 RINGKASAN DATA EKONOMI BULANAN")
print("   Range: Oktober 2025 - April 2026")
print("="*80)
print(df_sheet1.to_string(index=False))

print("\n💡 Keterangan:")
print("   - Data kurs berasal dari JISDOR (rata-rata bulanan)")
print("   - Inflasi Malang: Data real Jan 2025 - Mar 2026")
print("   - Inflasi Nasional: Data real Aug 2025 - Mar 2026")
print("   - Missing value diisi dengan interpolasi linear")