import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Game Sales Prediction App",
    page_icon="🎮",
    layout="wide"
)

# --- JUDUL & INTRO ---
st.title("🎮 Dashboard Prediksi Penjualan Video Game")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning** untuk memprediksi penjualan global video game 
berdasarkan skor review (Kritikus & User), Genre, Platform, dan Tahun Rilis.
""")

# --- FUNGSI LOAD DATA & CLEANING ---
@st.cache_data
def load_and_clean_data(uploaded_file):
    try:
        # 1. Baca Data Mentah
        df = pd.read_csv(uploaded_file)
        
        # Simpan bentuk data awal (Baris, Kolom)
        initial_shape = df.shape
        
        # Validasi Kolom
        required_columns = ['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 
                            'Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"File CSV kekurangan kolom: {', '.join(missing_cols)}")
            return None, None

        # Cleaning 'tbd' pada User_Score
        df['User_Score'] = df['User_Score'].replace('tbd', np.nan)
        df['User_Score'] = df['User_Score'].astype(float)
        
        # Drop Missing Values pada kolom penting
        cols_critical = ['Global_Sales', 'Critic_Score', 'User_Score', 'Year_of_Release', 'Publisher']
        df_clean = df.dropna(subset=cols_critical).copy()
        
        # Ubah Tahun jadi int
        df_clean['Year_of_Release'] = df_clean['Year_of_Release'].astype(int)
        
        # Log Transform untuk Target (Mengurangi efek outlier/skewness)
        df_clean['Global_Sales_Log'] = np.log1p(df_clean['Global_Sales'])
        
        return df_clean, initial_shape

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        return None, None

# --- SIDEBAR: UPLOAD & INFO ---
st.sidebar.header("📂 Upload Data")
st.sidebar.info("Gunakan dataset **Video Game Sales with Ratings** dari Kaggle.")
uploaded_file = st.sidebar.file_uploader("Upload 'Video_Games_Sales_as_at_22_Dec_2016.csv'", type=["csv"])

# --- LOGIKA UTAMA ---
if uploaded_file is not None:
    # Load data dan ambil ukuran awal
    df, initial_shape = load_and_clean_data(uploaded_file)
    
    if df is not None:
        # Tabs Navigasi
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 EDA & Visualisasi", "🤖 Model Performance", "🚀 Prediksi Baru"])

        # ==========================
        # TAB 1: DATA OVERVIEW
        # ==========================
        with tab1:
            st.subheader("Laporan Pembersihan Data")
            
            # --- FITUR BARU: TAMPILKAN SEBELUM & SESUDAH ---
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Total Data Awal (Raw)", f"{initial_shape[0]} Baris")
            with col_info2:
                st.metric("Total Data Bersih", f"{df.shape[0]} Baris")
            with col_info3:
                # Hitung selisih
                diff = initial_shape[0] - df.shape[0]
                st.metric("Data Dibuang (Missing Value)", f"{diff} Baris", delta_color="inverse")
            
            st.info("⚠️ Data berkurang signifikan karena kita membuang baris yang tidak memiliki **Skor Review** (Critic/User Score), agar model prediksi akurat.")
            
            st.divider()
            
            st.subheader("Cuplikan Data Bersih")
            st.dataframe(df.head(10))

            # Fitur Download
            csv_clean = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Dataset Bersih (CSV)", csv_clean, 'video_games_clean.csv', 'text/csv')
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.subheader("Statistik Numerik")
                st.write(df.describe())
            with col_d2:
                st.subheader("Statistik Kategorikal")
                st.write(df.describe(include=['O']))

        # ==========================
        # TAB 2: EDA & VISUALISASI
        # ==========================
        with tab2:
            st.markdown("### Exploratory Data Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("1. Distribusi Penjualan")
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                sns.histplot(df['Global_Sales'], kde=True, ax=ax[0], color='blue')
                ax[0].set_title("Sales Asli (Skewed)")
                sns.histplot(df['Global_Sales_Log'], kde=True, ax=ax[1], color='green')
                ax[1].set_title("Sales Log Transformed (Normal)")
                st.pyplot(fig)

            with col2:
                st.subheader("2. Korelasi Fitur")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                numeric_df = df.select_dtypes(include=[np.number])
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
                st.pyplot(fig2)

            st.divider()

            # --- 3 Grafik Tambahan ---
            st.subheader("3. Analisis Kategori & Tren")

            # Grafik A: Countplot Platform
            st.markdown("#### A. Jumlah Game per Platform (Top 15)")
            fig_plat, ax_plat = plt.subplots(figsize=(12, 5))
            top_platforms = df['Platform'].value_counts().iloc[:15].index
            sns.countplot(data=df, x='Platform', order=top_platforms, palette='viridis', ax=ax_plat)
            ax_plat.set_xticklabels(ax_plat.get_xticklabels(), rotation=45)
            st.pyplot(fig_plat)

            # Grafik B: Countplot Genre
            st.markdown("#### B. Jumlah Game per Genre")
            fig_gen, ax_gen = plt.subplots(figsize=(12, 5))
            sns.countplot(data=df, x='Genre', order=df['Genre'].value_counts().index, palette='magma', ax=ax_gen)
            ax_gen.set_xticklabels(ax_gen.get_xticklabels(), rotation=45)
            st.pyplot(fig_gen)

            # Grafik C: Scatterplot
            st.markdown("#### C. Hubungan Skor Kritikus vs Penjualan Global")
            fig_scat, ax_scat = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='Critic_Score', y='Global_Sales', hue='Genre', alpha=0.6, ax=ax_scat)
            ax_scat.set_xlabel("Critic Score")
            ax_scat.set_ylabel("Global Sales (Millions)")
            st.pyplot(fig_scat)

        # ==========================
        # PREPROCESSING & TRAINING
        # ==========================
        # Hapus kolom yang tidak dipakai untuk training
        cols_to_drop = ['Name', 'Developer', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Rating']
        # Pastikan kolom ada sebelum drop
        existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        
        X_raw = df.drop(columns=existing_cols_to_drop + ['Global_Sales_Log'])
        y = df['Global_Sales_Log']

        # One-Hot Encoding
        X = pd.get_dummies(X_raw, drop_first=True)
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        @st.cache_resource
        def train_models(X_train, y_train):
            model_lr = LinearRegression()
            model_lr.fit(X_train, y_train)
            
            model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model_rf.fit(X_train, y_train)
            return model_lr, model_rf

        with st.spinner('Sedang melatih model Machine Learning...'):
            model_lr, model_rf = train_models(X_train, y_train)

        # ==========================
        # TAB 3: MODEL PERFORMANCE
        # ==========================
        with tab3:
            st.subheader("Komparasi Model")
            st.write("Distribusi Data:")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Dataset", f"{df.shape[0]} Baris")
            c2.metric("Data Latih (80%)", f"{X_train.shape[0]} Baris")
            c3.metric("Data Uji (20%)", f"{X_test.shape[0]} Baris")
            
            st.divider()

            # Predict
            y_pred_lr = model_lr.predict(X_test)
            y_pred_rf = model_rf.predict(X_test)
            
            # Inverse Log
            y_test_real = np.expm1(y_test)
            y_pred_lr_real = np.expm1(y_pred_lr)
            y_pred_rf_real = np.expm1(y_pred_rf)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("### 📊 Linear Regression")
                st.metric("R2 Score", f"{r2_score(y_test_real, y_pred_lr_real):.4f}")
                st.metric("RMSE (Juta Unit)", f"{np.sqrt(mean_squared_error(y_test_real, y_pred_lr_real)):.4f}")
                st.metric("Jumlah Prediksi", f"{len(y_pred_lr)} Data")
            
            with col_m2:
                st.markdown("### 🌲 Random Forest (Terbaik)")
                st.metric("R2 Score", f"{r2_score(y_test_real, y_pred_rf_real):.4f}")
                st.metric("RMSE (Juta Unit)", f"{np.sqrt(mean_squared_error(y_test_real, y_pred_rf_real)):.4f}")
                st.metric("Jumlah Prediksi", f"{len(y_pred_rf)} Data")

            st.divider()
            
            # --- BAGIAN INSIGHT MODEL (BARU) ---
            st.header("🔍 Bedah Model (Model Insights)")
            
            # 1. Insight Linear Regression (Koefisien)
            st.subheader("1. Apa yang Dilihat Linear Regression? (Arah Hubungan)")
            st.write("""
            Grafik ini menunjukkan **Koefisien**, yaitu arah dan kekuatan hubungan setiap fitur terhadap penjualan.
            * **Warna Hijau (Positif):** Fitur ini meningkatkan prediksi penjualan.
            * **Warna Merah (Negatif):** Fitur ini menurunkan prediksi penjualan.
            """)
            
            # Ambil Koefisien & Nama Fitur
            coef_series = pd.Series(model_lr.coef_, index=X.columns)
            
            # Ambil Top 10 Fitur dengan Dampak Terbesar (Absolut)
            top_indices = coef_series.abs().sort_values(ascending=False).head(10).index
            top_coefs = coef_series[top_indices].sort_values() # Urutkan nilai untuk plotting yang rapi
            
            # Siapkan Data Plot
            coef_df = pd.DataFrame({'Feature': top_coefs.index, 'Coefficient': top_coefs.values})
            coef_df['Effect'] = ['Positif' if x > 0 else 'Negatif' for x in coef_df['Coefficient']]
            
            # Plot
            fig_lr, ax_lr = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Coefficient', y='Feature', data=coef_df, hue='Effect', 
                        palette={'Positif': 'green', 'Negatif': 'red'}, ax=ax_lr)
            ax_lr.axvline(0, color='black', linewidth=0.8) # Garis tengah 0
            ax_lr.set_title("Top 10 Koefisien Linear Regression Terbesar")
            st.pyplot(fig_lr)
            
            st.divider()

            # 2. Feature Importance Random Forest
            st.subheader("2. Apa yang Dilihat Random Forest? (Tingkat Kepentingan)")
            st.write("""
            Grafik ini menunjukkan fitur mana yang paling sering digunakan oleh Random Forest untuk membuat keputusan.
            Model ini tidak melihat "Positif/Negatif", tetapi melihat "Seberapa Berguna" fitur tersebut.
            """)
            
            importances = model_rf.feature_importances_
            feature_names = X.columns
            
            rf_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(10)
            
            fig_rf, ax_rf = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Importance', y='Feature', data=rf_importance_df, palette='viridis', ax=ax_rf)
            ax_rf.set_title("Top 10 Fitur Paling Berpengaruh (Random Forest)")
            st.pyplot(fig_rf)

        # ==========================
        # TAB 4: PREDIKSI (SIMULASI)
        # ==========================
        with tab4:
            st.subheader("Simulasi Game Baru")
            
            col_in1, col_in2, col_in3 = st.columns(3)
            with col_in1:
                p_platform = st.selectbox("Platform", sorted(df['Platform'].unique()))
                p_genre = st.selectbox("Genre", sorted(df['Genre'].unique()))
                p_year = st.number_input("Tahun Rilis", 1980, 2030, 2016)
            with col_in2:
                p_critic_score = st.slider("Critic Score (0-100)", 0, 100, 85)
                p_critic_count = st.number_input("Jumlah Kritikus", 0, value=50)
            with col_in3:
                p_user_score = st.slider("User Score (0-10)", 0.0, 10.0, 8.0)
                p_user_count = st.number_input("Jumlah User Review", 0, value=500)
            
            if st.button("Prediksi Penjualan (Random Forest)"):
                input_data = pd.DataFrame(columns=X_train.columns)
                input_data.loc[0] = 0
                
                input_data.loc[0, 'Year_of_Release'] = p_year
                input_data.loc[0, 'Critic_Score'] = p_critic_score
                input_data.loc[0, 'User_Score'] = p_user_score
                input_data.loc[0, 'Critic_Count'] = p_critic_count
                input_data.loc[0, 'User_Count'] = p_user_count
                
                if f"Platform_{p_platform}" in input_data.columns:
                    input_data.loc[0, f"Platform_{p_platform}"] = 1
                if f"Genre_{p_genre}" in input_data.columns:
                    input_data.loc[0, f"Genre_{p_genre}"] = 1
                    
                pred_real = np.expm1(model_rf.predict(input_data)[0])
                
                st.balloons()
                st.success(f"💰 Prediksi Penjualan Global: **{pred_real:.2f} Juta Unit**")

    else:
        st.warning("Data tidak valid atau terjadi error.")

else:
    st.info("👋 Silakan upload file CSV dataset di sidebar sebelah kiri.")