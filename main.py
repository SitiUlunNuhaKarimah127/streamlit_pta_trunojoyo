# Modul Library
import streamlit as st
import pandas as pd
import joblib

st.write("""
         Nama   : Siti Ulun Nuha Karimah\n
         NIM    : 200411100127\n
         Kelas  : Pencarian Dan Penambangan Web A\n
         """)
st.title('PTA Universitas Trunojoyo Madura')
deskripsi, dataset, preprocessing, vsm, lda, implementation = st.tabs(
    ["Info", "Dataset", "Preprocessing", "VSM", "LDA", "Implementation"])

with deskripsi:
    st.write("""
             Portal Artikel Tugas Akhir Universitas Trunojoyo Madura adalah platform daring yang menyediakan akses dan informasi mengenai karya akhir mahasiswa-mahasiswa yang telah menyelesaikan program studi di Universitas Trunojoyo Madura. 
             Portal ini memberikan akses kepada pengguna untuk menelusuri, membaca, dan memahami berbagai artikel, skripsi, tesis, dan disertasi yang telah dihasilkan oleh mahasiswa sebagai hasil penelitian dan kajian mereka selama menempuh 
             pendidikan di universitas tersebut. Tujuan utama dari portal ini adalah untuk mempermudah aksesibilitas, penyebarluasan, dan pemanfaatan karya akhir mahasiswa sebagai sumber inspirasi, penelitian, dan referensi bagi mahasiswa, 
             akademisi, dan masyarakat luas.
             """)

with dataset:
    st.write("""""")
    ("""
     """)

    st.write("""
             Keterangan Kolom :
             1. Judul
             2. Penulis
             3. Dosen Pembimbing I
             4. Dosen Pembimbing II
             5. Abstrak
             6. Label 
             """)
    st.write("""
             Di dalam dataset terdapat 858 data dan 6 kolom
             """)
    df = pd.read_csv('data/crawling_pta_labeled.csv')
    st.write(df)

with preprocessing:
    st.write("pada tahap ini data dilakukan preprocessing dengan menggunakan beberapa tahap di antaranya")

    st.write("""
             1. Cleaning
             2. Tokenizing
             3. Stopword Removal
             4. Stemming (Opsional)""")
    st.write("berikut hasil dari proses preprocesing:")
    df = pd.read_csv("data/data_preprocessing.csv")
    st.write(df)

with vsm:
    st.write("pada tahap ini dataset hasil preprocessing dilakukan prsoses ekstraksi fitur dengan menggunakan beberapa metode di antara nya:")
    st.write("""
             1. One Hot Encoder
             2. Term Frequency
             3. TF IDF
             4. Logarithm Frequency
             """)
    st.write("berikut adalah hasil dari masing-masing metode:")
    onehot, tf_idf, tf, lf = st.tabs(['One Hot Encoding', 'TF IDF',
                                      'Term Frequensi', 'Logaritm Frequency'])

    with onehot:
        df_onehot = pd.read_csv("data/OneHotEncoder.csv")
        st.write(df_onehot)

    with tf_idf:
        df_tf_idf = pd.read_csv("data/TF IDF.csv")
        st.write(df_tf_idf)

    with tf:
        df_tf = pd.read_csv("data/Term Frequensi.csv")
        st.write(df_tf)

    with lf:
        df_lf = pd.read_csv("data/Logarithm Frequensi.csv")
        st.write(df_lf)

with lda:
    st.write("Pada proses ini merupakan pengecilan dimensi dari feature atau biasanya di sebut dengan reduksi dimensi dengan metode yang digunakan adalah LDA (Latent Dirichlet Allocation), dimana dari feature yang aslinya ada 10 atau lebih di jadikan menjadi 3-n bisa disesuaikan")
    st.write("hasil reduksi dimensi :")
    df_lda = pd.read_csv('data/reduksi dimensi.csv')
    st.write(df_lda)

with implementation:

    abstrak = st.text_input("Masukkan Abstrak")
    model = joblib.load("model/NB.pkl")
    lda = joblib.load("model/LDA.pkl")
    vectorizer = joblib.load("model/tf_vectorizer.pkl")
    button = st.button("Klasifikasikan")

    if button:
        x_new = vectorizer.transform([abstrak])
        lda_x = lda.fit_transform(x_new)
        predictions = model.predict(lda_x)
        st.write("Abstrak termasuk dalam Kategori : ", predictions[0])
