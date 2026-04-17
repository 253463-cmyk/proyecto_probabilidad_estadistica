import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="App de Probabilidad - UP Chiapas", layout="wide")

# --- MENÚ LATERAL (SIDEBAR) ---
# Aquí es donde aparecerán las "rayitas" para abrir/cerrar opciones
with st.sidebar:
    st.title("⚙️ Menú de Configuración")
    st.markdown("---")
    
    st.header("1. Origen de Datos")
    opcion_carga = st.radio("Seleccionar:", ("Subir CSV", "Generación Sintética"))
    
    df = None
    if opcion_carga == "Subir CSV":
        archivo = st.file_uploader("Carga tu archivo .csv", type=["csv"])
        if archivo:
            df = pd.read_csv(archivo)
    else:
        n = st.number_input("Muestra (n)", min_value=30, value=100)
        mu = st.number_input("Media (μ)", value=0.0)
        sigma = st.number_input("Desviación (σ)", min_value=0.1, value=1.0)
        if st.button("Generar Datos"):
            datos = np.random.normal(mu, sigma, n)
            st.session_state['df_sintetico'] = pd.DataFrame(datos, columns=["Variable_Sintetica"])
        if 'df_sintetico' in st.session_state:
            df = st.session_state['df_sintetico']

    # Solo mostramos la configuración de la prueba si hay datos cargados
    if df is not None:
        st.markdown("---")
        st.header("2. Parámetros Z-Test")
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        columna = st.selectbox("Variable:", columnas_numericas)
        h0_valor = st.number_input("H₀ (Media hipotética)", value=float(df[columna].mean()))
        tipo_cola = st.selectbox("Prueba (H₁)", ["Bilateral (≠)", "Cola Derecha (>)", "Cola Izquierda (<)"])
        alpha = st.select_slider("Significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)

# --- ÁREA PRINCIPAL (RESULTADOS) ---
st.title("📊 Proyecto de Probabilidad y Estadística")
st.markdown("**Desarrollado por:** Elizabeth Escobar")

if df is not None:
    # Cálculos Estadísticos
    datos = df[columna].dropna()
    n_size, media_m, desv_p = len(datos), datos.mean(), datos.std()
    z_stat = (media_m - h0_valor) / (desv_p / np.sqrt(n_size))

    if tipo_cola == "Bilateral (≠)":
        p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif tipo_cola == "Cola Derecha (>)":
        p_value_z = 1 - stats.norm.cdf(z_stat)
    else:
        p_value_z = stats.norm.cdf(z_stat)

    # Organización de la pantalla principal en dos columnas
    col_izq, col_der = st.columns([1, 1.2])

    with col_izq:
        st.subheader("📝 Resultados de la Prueba")
        # Métricas más grandes y visibles
        st.metric(label="Estadístico Z", value=f"{z_stat:.4f}")
        st.metric(label="P-Value", value=f"{p_value_z:.4f}")
        
        if p_value_z < alpha:
            st.error("**Veredicto: Rechazar H₀**")
        else:
            st.success("**Veredicto: No Rechazar H₀**")
        
        # Prueba de Normalidad integrada aquí mismo
        st.markdown("---")
        stat_w, p_val_shapiro = stats.shapiro(datos)
        st.write(f"**Normalidad (Shapiro-Wilk):** `{p_val_shapiro:.4f}`")
        if p_val_shapiro > 0.05:
            st.caption("✅ Distribución Normal detectada.")
        else:
            st.caption("⚠️ Los datos no parecen seguir una distribución normal.")

    with col_der:
        st.subheader("📈 Visualización de Datos")
        fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True, 
                                              gridspec_kw={"height_ratios": (.7, .3)})
        sns.histplot(datos, kde=True, ax=ax_hist, color="skyblue", edgecolor="black")
        sns.boxplot(x=datos, ax=ax_box, color="lightgreen", linewidth=1)
        plt.tight_layout()
        st.pyplot(fig)

    # --- MÓDULO DE IA ---
    if api_key:
        st.markdown("---")
        if st.button("🤖 Generar Análisis con IA"):
            # (Aquí va tu lógica de Gemini que ya tienes)
            st.info("Generando resumen...")
else:
    st.info("👈 Abre el menú lateral y carga un archivo para comenzar el análisis.")