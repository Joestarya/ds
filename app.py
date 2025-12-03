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

# Konfigurasi Halaman
st.set_page_config(
    page_title="Game Sales Prediction App",
    page_icon="🎮",
    layout="wide"
)

# Judul
st.title("🎮 Dashboard Prediksi Penjualan Video Game")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning** untuk memprediksi penjualan global video game 
berdasarkan skor review (Kritikus & User), Genre, Platform, dan Tahun Rilis.
""")

# --- 1. LOAD DATA & CLEANING (CACHED) ---
@st.cache_data
def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Cleaning 'tbd'
    df['User_Score'] = df['User_Score'].replace('tbd', np.nan)
    df['User_Score'] = df['User_Score'].astype(float)
    
    # Drop Missing Values pada kolom penting
    cols_critical = ['Global_Sales', 'Critic_Score', 'User_Score', 'Year_of_Release', 'Publisher']
    df_clean = df.dropna(subset=cols_critical).copy()
    
    # Ubah Tahun jadi int
    df_clean['Year_of_Release'] = df_clean['Year_of_Release'].astype(int)
    
    # Log Transform untuk Target
    df_clean['Global_Sales_Log'] = np.log1p(df_clean['Global_Sales'])
    
    return df_clean

# --- SIDEBAR: UPLOAD ---
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file 'Video_Games_Sales_as_at_22_Dec_2016.csv'", type=["csv"])

if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 EDA", "🤖 Model Performance", "🚀 Prediksi Baru"])

    # ==========================
    # TAB 1: DATA OVERVIEW
    # ==========================
    with tab1:
        st.subheader("Cuplikan Data Bersih")
        st.write(f"Jumlah Baris Data: **{df.shape[0]}**")
        st.dataframe(df.head(10))
        
        st.subheader("Statistik Deskriptif")
        st.write(df.describe())

    # ==========================
    # TAB 2: EDA
    # ==========================
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi Penjualan (Target)")
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            
            # Asli
            sns.histplot(df['Global_Sales'], kde=True, ax=ax[0], color='blue')
            ax[0].set_title("Sales Asli (Skewed)")
            
            # Log
            sns.histplot(df['Global_Sales_Log'], kde=True, ax=ax[1], color='green')
            ax[1].set_title("Sales Log Transformed (Normal)")
            
            st.pyplot(fig)
            st.info("Kita menggunakan **Log Transformation** agar model lebih akurat memprediksi data yang timpang.")

        with col2:
            st.subheader("Korelasi Fitur")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            numeric_df = df.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
            st.pyplot(fig2)
            
            # Bukti Statistik
            r, p = stats.pearsonr(df['Critic_Score'], df['Global_Sales'])
            st.write(f"**Korelasi Pearson (Critic Score vs Sales):** `{r:.3f}`")

    # ==========================
    # PREPROCESSING & TRAINING
    # ==========================
    # Hapus kolom bocoran & identifier
    cols_to_drop = ['Name', 'Developer', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    X_raw = df.drop(columns=cols_to_drop + ['Global_Sales_Log'])
    y = df['Global_Sales_Log']

    # Encoding
    X = pd.get_dummies(X_raw, drop_first=True)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model (Cached)
    @st.cache_resource
    def train_models(X_train, y_train):
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        model_rf.fit(X_train, y_train)
        
        return model_lr, model_rf

    model_lr, model_rf = train_models(X_train, y_train)

    # ==========================
    # TAB 3: MODEL PERFORMANCE
    # ==========================
    with tab3:
        st.subheader("Komparasi Model")
        
        # Predict
        y_pred_lr = model_lr.predict(X_test)
        y_pred_rf = model_rf.predict(X_test)
        
        # Inverse Log
        y_test_real = np.expm1(y_test)
        y_pred_lr_real = np.expm1(y_pred_lr)
        y_pred_rf_real = np.expm1(y_pred_rf)
        
        # Metrics
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("### Linear Regression")
            st.metric("R2 Score", f"{r2_score(y_test_real, y_pred_lr_real):.4f}")
            st.metric("RMSE (Juta Unit)", f"{np.sqrt(mean_squared_error(y_test_real, y_pred_lr_real)):.4f}")
            
        with col_m2:
            st.markdown("### Random Forest (Terbaik)")
            st.metric("R2 Score", f"{r2_score(y_test_real, y_pred_rf_real):.4f}")
            st.metric("RMSE (Juta Unit)", f"{np.sqrt(mean_squared_error(y_test_real, y_pred_rf_real)):.4f}")

        st.divider()
        
        # Business Insight
        st.subheader("💡 Insight Bisnis: Critic vs User")
        coef = pd.Series(model_lr.coef_, index=X.columns)
        
        c_impact = coef.get('Critic_Score', 0)
        u_impact = coef.get('User_Score', 0)
        
        col_i1, col_i2 = st.columns(2)
        col_i1.info(f"Dampak Critic Score: {c_impact:.5f}")
        col_i2.info(f"Dampak User Score: {u_impact:.5f}")
        
        if c_impact > u_impact:
            st.success("✅ **Kesimpulan:** Review dari Kritikus/Media lebih berpengaruh meningkatkan penjualan dibandingkan User.")
        else:
            st.success("✅ **Kesimpulan:** Review dari User lebih berpengaruh.")

    # ==========================
    # TAB 4: PREDIKSI (SIMULASI)
    # ==========================
    with tab4:
        st.subheader("Simulasi Game Baru")
        st.write("Masukkan parameter game yang akan dirilis:")
        
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            # Dropdown mengambil nilai unik dari data asli
            platforms = sorted(df['Platform'].unique())
            genres = sorted(df['Genre'].unique())
            
            p_platform = st.selectbox("Platform", platforms)
            p_genre = st.selectbox("Genre", genres)
            p_year = st.number_input("Tahun Rilis", min_value=1980, max_value=2030, value=2016)
            
        with col_in2:
            p_critic_score = st.slider("Critic Score (0-100)", 0, 100, 85)
            p_critic_count = st.number_input("Jumlah Kritikus", min_value=0, value=50)
            
        with col_in3:
            p_user_score = st.slider("User Score (0-10)", 0.0, 10.0, 8.0)
            p_user_count = st.number_input("Jumlah User Review", min_value=0, value=500)
            
        if st.button("Prediksi Penjualan"):
            # 1. Buat DataFrame Input Kosong sesuai kolom training
            input_data = pd.DataFrame(columns=X_train.columns)
            input_data.loc[0] = 0  # Isi 0 semua
            
            # 2. Isi Data Numerik
            input_data.loc[0, 'Year_of_Release'] = p_year
            input_data.loc[0, 'Critic_Score'] = p_critic_score
            input_data.loc[0, 'User_Score'] = p_user_score
            input_data.loc[0, 'Critic_Count'] = p_critic_count
            input_data.loc[0, 'User_Count'] = p_user_count
            
            # 3. Handle One-Hot Encoding Manual
            # Format kolom di X_train biasanya: 'Platform_PS4', 'Genre_Action'
            col_plat = f"Platform_{p_platform}"
            col_gen = f"Genre_{p_genre}"
            
            if col_plat in input_data.columns:
                input_data.loc[0, col_plat] = 1
            if col_gen in input_data.columns:
                input_data.loc[0, col_gen] = 1
                
            # 4. Prediksi (Random Forest)
            pred_log = model_rf.predict(input_data)[0]
            pred_real = np.expm1(pred_log)
            
            st.balloons()
            st.success(f"💰 Prediksi Penjualan Global: **{pred_real:.2f} Juta Unit**")
            
            # Contextual info
            if pred_real > 1.0:
                st.write("🎉 Ini potensi **Blockbuster**!")
            elif pred_real < 0.1:
                st.write("⚠️ Penjualan diprediksi rendah (Niche market).")

else:
    st.info("Silakan upload file CSV dataset di sidebar sebelah kiri untuk memulai.")
    st.markdown("""
    **Format File:** Dataset harus memiliki kolom: `Name`, `Platform`, `Year_of_Release`, `Genre`, `Publisher`, `Global_Sales`, `Critic_Score`, `Critic_Count`, `User_Score`, `User_Count`.
    """)