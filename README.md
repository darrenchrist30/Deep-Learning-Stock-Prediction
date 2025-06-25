# Stock Price Prediction Web Application

Aplikasi web prediksi harga saham menggunakan model LSTM pre-trained.

## Tentang Aplikasi

Aplikasi ini menggunakan model LSTM (Long Short-Term Memory) yang telah dilatih untuk memprediksi harga saham. Aplikasi ini memungkinkan pengguna untuk:

1. Mengunggah model LSTM yang sudah dilatih (file .h5)
2. Mengunggah scaler yang sesuai (file .pkl)
3. Mengunggah data historis saham (file .csv)
4. Melihat visualisasi prediksi harga saham (historis dan masa depan)

## Prasyarat

Pastikan Anda memiliki:

1. Python 3.8 atau lebih tinggi
2. XAMPP untuk menjalankan web server
3. Model LSTM terlatih (.h5) dan file scaler (.pkl)
4. Data saham historis dalam format CSV dengan kolom 'Date' dan 'Close'

## Instalasi

1. Letakkan semua file di direktori `c:\xampp\htdocs\proyek-deep-learning`
2. Install paket Python yang dibutuhkan:
   ```
   pip install -r requirements.txt
   ```

## Menjalankan Aplikasi

1. Buka terminal dan masuk ke direktori proyek:

   ```
   cd c:\xampp\htdocs\proyek-deep-learning
   ```

2. Jalankan aplikasi Flask:

   ```
   python app.py
   ```

3. Buka browser dan akses aplikasi di `http://localhost:5000`

## Penggunaan

### Mengunggah Model dan Scaler

1. Pada halaman utama, cari bagian "Upload Model"
2. Unggah file model LSTM (.h5) dan file scaler (.pkl)
3. Klik "Upload Model"
4. Status model akan berubah menjadi "Model is loaded and ready to use"

### Membuat Prediksi

1. Siapkan file CSV dengan minimal kolom 'Date' dan 'Close'
2. Pada halaman utama, cari bagian "Upload Stock Data"
3. Unggah file CSV yang sudah disiapkan
4. Klik "Upload and Predict"
5. Hasil prediksi akan ditampilkan dalam bentuk grafik dan tabel

## Format File CSV

File CSV untuk data saham harus memiliki format sebagai berikut:

```
Date,Open,High,Low,Close,Volume
2020-01-01,100.0,105.0,95.0,102.0,10000
2020-01-02,102.0,107.0,100.0,105.0,12000
...
```

Kolom yang diperlukan:

- `Date`: Tanggal dalam format YYYY-MM-DD
- `Close`: Harga penutupan saham

## Catatan Penting

1. Model LSTM Anda dilatih untuk melakukan prediksi berdasarkan pola data historis tertentu. Pastikan data yang Anda unggah memiliki pola dan format yang mirip dengan data pelatihan.
2. Diperlukan minimal 60+ baris data untuk membuat prediksi.
3. File scaler (.pkl) harus cocok dengan model LSTM (.h5) yang diunggah untuk memastikan transformasi data yang tepat.
4. Prediksi masa depan merupakan estimasi berdasarkan pola historis dan tidak menjamin akurasi dalam kondisi pasar nyata.

## Solusi Masalah Umum

1. **Model tidak dimuat**: Pastikan format file model (.h5) dan scaler (.pkl) sesuai dan kompatibel.
2. **Error saat prediksi**: Periksa format data CSV. Pastikan ada kolom 'Date' dan 'Close'.
3. **Hasil prediksi tidak akurat**: Pastikan data saham yang digunakan memiliki pola yang mirip dengan data pelatihan.
4. **Aplikasi tidak berjalan**: Periksa instalasi paket Python dan pastikan versi yang sesuai.
