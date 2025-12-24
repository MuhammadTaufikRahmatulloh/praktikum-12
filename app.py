import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="TA-12 | EDA Time Series",
    layout="wide"
)

st.title("EDA Time Series Dashboard")
st.caption("Tugas Akhir 12 · Pemodelan dan Simulasi")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_weather.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

df = load_data()

# ================= SIDEBAR =================
st.sidebar.header("Pengaturan")
variable = st.sidebar.selectbox(
    "Pilih Variabel",
    df.columns
)

# ================= OVERVIEW =================
st.subheader("Overview")
st.write(
    "Dashboard ini menampilkan hasil Exploratory Data Analysis (EDA) "
    "pada dataset cuaca berbasis time series."
)

# ================= VISUAL INSPECTION =================
st.subheader("Visual Inspection")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Data Asli (Resolusi Tinggi)**")
    fig, ax = plt.subplots()
    ax.plot(df[variable])
    ax.set_xlabel("Waktu")
    ax.set_ylabel(variable)
    st.pyplot(fig)

with col2:
    st.markdown("**Data Resampling Bulanan**")
    df_monthly = df[variable].resample('M').mean()
    fig, ax = plt.subplots()
    ax.plot(df_monthly)
    ax.set_xlabel("Waktu")
    ax.set_ylabel(variable)
    st.pyplot(fig)

# ================= DECOMPOSITION =================
st.subheader("Time Series Decomposition")

df_daily = df[variable].resample('D').mean()

decomp = seasonal_decompose(
    df_daily.dropna(),
    model='additive',
    period=7
)

st.pyplot(decomp.plot())

# ================= STATIONARITY =================
st.subheader("Stationarity Check")

rolling_mean = df_monthly.rolling(12).mean()
rolling_std = df_monthly.rolling(12).std()

fig, ax = plt.subplots()
ax.plot(df_monthly, label="Data Asli")
ax.plot(rolling_mean, label="Rolling Mean")
ax.plot(rolling_std, label="Rolling Std")
ax.legend()
st.pyplot(fig)

st.success("Analisis EDA Time Series berhasil ditampilkan.")
