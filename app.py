import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración básica de la página
st.set_page_config(page_title="App de Probabilidad - UP Chiapas", layout="wide")

st.title("📊 Proyecto de Probabilidad y Estadística")
st.markdown(f"**Desarrollado por:** Elizabeth Escobar")
st.markdown("---")

# --- MÓDULO 1: CARGA DE DATOS ---
st.header("1. Carga de Datos")

opcion_carga = st.radio("Selecciona el origen de los datos:", 
                         ("Subir CSV", "Generación Sintética (Normal)"))

df = None

if opcion_carga == "Subir CSV":
    archivo = st.file_uploader("Carga tu archivo .csv", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        st.success("¡Archivo cargado con éxito!")

else:
    n = st.number_input("Tamaño de muestra (n)", min_value=30, value=100)
    mu = st.number_input("Media hipotética (μ)", value=0.0)
    sigma = st.number_input("Desviación estándar (σ)", min_value=0.1, value=1.0)
    
    if st.button("Generar Datos Aleatorios"):
        datos = np.random.normal(mu, sigma, n)
        df = pd.DataFrame(datos, columns=["Variable_Sintetica"])
        st.info("Datos generados exitosamente.")

# --- MÓDULO 2: VISUALIZACIÓN ---
if df is not None:
    st.markdown("---")
    st.header("2. Visualización de Distribuciones")
    
    columna = st.selectbox("Selecciona la variable a analizar:", df.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Histograma (Distribución)")
        fig, ax = plt.subplots()
        sns.histplot(df[columna], kde=True, ax=ax, color="skyblue")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Estadísticos Descriptivos")
        st.write(df[columna].describe())

    st.success("Fase 2 completada: Datos listos para análisis.")