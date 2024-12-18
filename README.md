# Laporan Proyek Machine Learning - Sistem Rekomendasi

![Gambar Banana](https://raw.githubusercontent.com/lusiaulia/anime-recommendation/refs/heads/main/dataset-cover.png)

## Domain Proyek 
Industri hiburan dan media terus dan semakin berkembang seiring dengan perkembangan zaman. Salah satu hiburan yang terkenal di seluruh dunia khas berasal dari negara Jepang yaitu anime, anime adalah animasi asal Jepang yang digambar dengan tangan maupun menggunakan teknologi komputer [WIkipedia,2024](https://id.wikipedia.org/wiki/Anime).
Keberhasilan anime dalam menjajaki industri hiburan dan bisa menghibur beragam orang di seluruh dunia, salah satunya disebabkan beragamnya jenis dan film anime baik yang diperuntukkan untuk anak-anak hingga dewasa, ataupun dari genre slice of life, comedy, romansa, youth dan lainnya. 
Pada percobaan ini, akan dilakukan percobaan untuk membuat rekomendasi anime berdasar kemiripan penonton dan hasil rating yang diberikan di platform myanimelist. Sehingga, akan memudahkan untuk penonton memilih anime yang akan ditonton sesuai dengan minatnya. 

## Business Understanding
### Problem Statements
- Apakah data-data yang tersedia bisa digunakan untuk membuat sistem rekomendasi judul anime kepada pengguna?
- Apakah hasil rekomendasi yang diberikan cukup relevan (seperti persamaan genre, atau rating)?

### Goals
- Mengetahui apakah data variabel yang tersedia dapat digunakan untuk sistem rekomendasi judul anime kepada pengguna.
- Mengetahui kerelevanan hasil rekomendasi dilihat dari genre, judul anime, atau rating output rekomendasi. 

### Solution Statements
- Menggunakan model machine learning untuk proses training rekomendasi anime. Model yang digunakan yaitu RecommenderNet dengan pendekatan rekomendasi user-based.

## Data Understanding
Data bersumber dari [Kaggle Anime Rating Dataset](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database). Terdiri dari 2 file yaitu anime.csv dan rating.csv. Tahapan dari data understanding meliputi Exploratory Data Analysis (EDA), yang kemudian diperoleh informasi variabel data dari 2 file yang digunakan yaitu : 

1. Variabel anime.csv

- anime_id : id anime dari myanimelist.
- name : judul anime
- genre : genre anime
- type : movie, TV, OVA, dll.
- episodes : banyak episode (1 jika movie).
- rating : rata-rata rating dari anime (skala 1-10)
- members : jumlah member dalam grup anime

2. Variabel rating.csv

- user_id : id user
- anime_id : anime yang dinilai/rate oleh user
- rating : rating skala 1-10 yang diberi oleh user (rating -1 jika sudah menonton tapi belum memberi rating)

File anime.csv berisi 12294 entri data di tiap variabel, sementara file rating.csv berisi 7813737 entri data tiap variabel.  Sementara untuk jumlah unik(tanpa duplikat) untuk id anime pada anime.csv sejumlah 12294 dan pada rating.csv sejumlah 11200. Nantinya akan digunakan data pada rating.csv untuk proses training model rekomendasi yang dibangun. 

Agar memudahkan dalam memahami data, akan coba dilihat jenis tipe anime yang ada di list data. Ternyata terdapat 6 jenis type anime yang ada yaitu tipe Movie, TV, OVA, Spesial, Music, dan ONA. Dengan tipe anime yang tayang di TV menjadi data paling banyak. Untuk rating sendiri divisualisasikan dengan histogram terlihat rata-rata berada di kisaran 6-7 untuk data nilai rating terbanyak.


## Data Preparation
Tahapan pengolahan data yang dilakukan meliputi proses yang mencakup : 
### Data Cleaning
Ingat pada file rating.csv untuk data rating terdapat nilai -1 untuk pengguna yang sudah menonton animenya tapi belum memberi rating. Data tersebut diputuskan untuk dihapus agar rating yang tersedia hanya berada di range rating (1-10). Akibatnya, jumlah id anime tersisa 9927 pada rating.csv dan banyak data menjadi 633.7241 data. 

#### Null Value & Data Duplikat
Perlu diperhatikan agar pada data tidak ada data kosong dan juga data yang duplikat, sehingga dilakukan penghapusan data kosong & duplikat pada kedua file. 

### Data Transform (Standarisasi) & Pembagian Data
Sebelum data digunakan untuk proses training model yang disini akan digunakan RecomenderNet dengan data rating.csv, terlebih dahulu data rating (variabel y/tujuan) akan diskalakan agar memudahkan yaitu berada di rentang 0-1. Kemudian dilakukan pembagian data train dan test, dengan train berisi 80% data dan test 20%. Lalu data x terdiri dari user_id dan anime_id, sementara data y yaitu rating.

## Modeling
Pada penelitian ini dibuat model yang biasa digunakan dalam rekomendasi, yaitu : 
### User-Based Collaborative Filtering
Data anime_df yang digunakan untuk proses membuat rekomendasi menggunakan pendekatan user-based dengan TF-IDF Vectorizer terlebih dahulu diubah menjadi bentuk matriks untuk dapat dilihat interaksi dari variabel (disini variabel name) dengan variabel yang dianggap penting dari algoritma. Kemudian akan dicari k-n data terdekat yang secara skor memiliki kemiripan tertinggi. 

### RecommederNet
Merupakan model rekomendasi yang digunakan untuk memberikan rekomendasi item kepada pengguna berdasarkan preferensi atau riwayat interaksi. Pada nodel ini digunakan embedding layers untuk merepresentasikan fitur dari pengguna dan item, kemudian digunakan dot product untuk menghitung kemiripan pengguna dengan item, juga terdapat fungsi loss untuk mengukur seberapa jauh prediksi model dari nilai sebenarnya. 


## Evaluation & Conclusion
- Proses evaluasi menggunakan RMSE, diperoleh nilai 0,37 dari proses data training, kemudian 0,32 untuk data test. 
- Dilihat dari grafik evaluasi, bertambahnya epoch malah menunjukkan RMSE yang semakin besar. Sehingga untuk selanjutnya bisa dilakukan uji coba pengurangan ukuran batch size, pengurangan data training dan penambahan epoch, penambahan layer model, dan proses hyperparameter tuning lainnya.
- Setelah dilakukan prediksi rekomendasi dengan nomor user 2, diperoleh top-5 anime dengan rating terbesar yang mirip dengan user 2 yaitu anime bergenre adventure, sci-fi dan shonen yang mendominasi. Untuk predicted rating yang dikeluarkan juga sangat mirip dari 1 anime ke anime lain yang direkomendasikan sehingga bisa diasumsikan bahwa data yang ada bisa digunakan untuk proses rekomendasi dan bisa merekomendasikan film yang memiliki type yang mirip. 