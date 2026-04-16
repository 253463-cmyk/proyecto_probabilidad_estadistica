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

# --- MÓDULO 3: INFERENCIA ESTADÍSTICA ---
    st.markdown("---")
    st.header("3. Inferencia: Calculadora de Probabilidades")
    
    # Extraemos media y desviación de la variable seleccionada
    media_obs = df[columna].mean()
    desv_obs = df[columna].std()
    
    st.write(f"Analizando variable: **{columna}**")
    st.latex(rf"\mu = {media_obs:.4f}, \quad \sigma = {desv_obs:.4f}")

    col_calc1, col_calc2 = st.columns(2)
    
    with col_calc1:
        st.subheader("Calcular P(X < x)")
        valor_x = st.number_input("Ingresa el valor de x:", value=float(media_obs))
        
        # Cálculo de Z y Probabilidad usando la librería scipy
        from scipy import stats
        z_score = (valor_x - media_obs) / desv_obs
        probabilidad = stats.norm.cdf(z_score)
        
        st.metric("Puntaje Z", f"{z_score:.4f}")
        st.metric("Probabilidad P(X < x)", f"{probabilidad:.4%}")
        
    with col_calc2:
        st.subheader("Gráfico de Probabilidad")
        fig, ax = plt.subplots()
        # Creamos la curva normal teórica
        x_plot = np.linspace(media_obs - 4*desv_obs, media_obs + 4*desv_obs, 100)
        y_plot = stats.norm.pdf(x_plot, media_obs, desv_obs)
        
        ax.plot(x_plot, y_plot, color="black")
        # Coloreamos el área bajo la curva solicitada
        ax.fill_between(x_plot, y_plot, where=(x_plot <= valor_x), color='orange', alpha=0.5)
        st.pyplot(fig)