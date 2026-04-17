import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats # ¡CORRECCIÓN 1: Importación necesaria para la estadística!
import os
from dotenv import load_dotenv

load_dotenv() # Carga las variables del archivo .env
api_key = os.getenv("GEMINI_API_KEY")

# Configuración básica de la página
st.set_page_config(page_title="App de Probabilidad - UP Chiapas", layout="wide")

st.title("📊 Proyecto de Probabilidad y Estadística")
st.markdown("**Desarrollado por:** Elizabeth Escobar")
st.markdown("---")

# --- MÓDULO 1: CARGA DE DATOS ---
st.header("Carga de Datos")

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
    
    # Botón para generar y guardar en el estado de la sesión
    if st.button("Generar Datos Aleatorios"):
        datos = np.random.normal(mu, sigma, n)
        st.session_state['df_sintetico'] = pd.DataFrame(datos, columns=["Variable_Sintetica"])
        st.success("¡Datos generados y guardados en memoria!")

    # Si ya existen datos en la memoria, los asignamos a 'df'
    if 'df_sintetico' in st.session_state:
        df = st.session_state['df_sintetico']

# --- VALIDACIÓN PRINCIPAL ---
# ¡CORRECCIÓN 3: Todo lo de abajo ahora está correctamente indentado dentro del if!
if df is not None:
    
# --- MÓDULO 3: PANEL DE CONTROL ESTADÍSTICO (VERSIÓN FINAL) ---
  if df is not None:
    st.markdown("---")
    st.header("Pruebas Estadísticas e Inferencia")
    
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if columnas_numericas:
        col_params, col_viz = st.columns([1, 1.2])
        
        with col_params:
            st.subheader("⚙️ Parámetros")
            columna = st.selectbox("Variable:", columnas_numericas)
            h0_valor = st.number_input("H₀ (Media hipotética)", value=float(df[columna].mean()))
            tipo_cola = st.selectbox("Prueba (H₁)", ["Bilateral (≠)", "Cola Derecha (>)", "Cola Izquierda (<)"])
            alpha = st.select_slider("Significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)
            
            # Cálculos
            datos = df[columna].dropna()
            n, media_m, desv_p = len(datos), datos.mean(), datos.std()
            z_stat = (media_m - h0_valor) / (desv_p / np.sqrt(n))

            # P-Value y Z-Crítico
            if tipo_cola == "Bilateral (≠)":
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                z_critico = stats.norm.ppf(1 - alpha/2)
            elif tipo_cola == "Cola Derecha (>)":
                p_value = 1 - stats.norm.cdf(z_stat)
                z_critico = stats.norm.ppf(1 - alpha)
            else:
                p_value = stats.norm.cdf(z_stat)
                z_critico = stats.norm.ppf(alpha)
            
            # --- VISUALIZACIÓN DE MÉTRICAS (MÁS GRANDES) ---
            st.markdown("---")
            m1, m2 = st.columns(2)
            # st.metric hace que los números resalten más que el texto simple
            m1.metric("Estadístico Z", f"{z_stat:.3f}")
            m2.metric("P-Value", f"{p_value:.4f}")
            
            if p_value < alpha:
                st.error(f"**Veredicto:** Rechazar H₀")
                st.caption(f"El estadístico cayó en la zona de rechazo (Z > {abs(z_critico):.2f})")
            else:
                st.success(f"**Veredicto:** No Rechazar H₀")
                st.caption(f"No hay evidencia suficiente para descartar H₀")

        with col_viz:
            st.subheader("📊 Visualización de Datos")
            # Gráficas pequeñas y compactas
            fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(6, 4), sharex=True, 
                                                  gridspec_kw={"height_ratios": (.7, .3)})
            
            sns.histplot(df[columna], kde=True, ax=ax_hist, color="skyblue", edgecolor="black")
            ax_hist.set_ylabel("Freq", fontsize=8)
            sns.boxplot(x=df[columna], ax=ax_box, color="lightgreen", linewidth=1)
            
            plt.tight_layout()
            st.pyplot(fig)
            
    else:
        st.warning("No hay columnas numéricas.")

    # --- MÓDULO 4: PRUEBA DE NORMALIDAD ---
    st.markdown("---")
    st.header("Validación: Prueba de Shapiro-Wilk")
    
    # Realizar la prueba estadística
    stat, p_value = stats.shapiro(df[columna])
    
    col_sh1, col_sh2 = st.columns(2)
    
    with col_sh1:
        st.metric("Estadístico W", f"{stat:.4f}")
        st.metric("P-Value", f"{p_value:.4f}")
        
    with col_sh2:
        st.subheader("Veredicto")
        if p_value > 0.05:
            st.success("✅ Los datos siguen una Distribución Normal (p > 0.05).")
            st.write("Es seguro confiar en los cálculos de la Fase 3.")
        else:
            st.warning("⚠️ Los datos NO siguen una Distribución Normal (p <= 0.05).")
            st.write("Los resultados de la calculadora podrían no ser exactos para este conjunto de datos.")

    st.info("Nota: La prueba de Shapiro-Wilk es la más potente para muestras menores a 5000 datos.")

    # --- MÓDULO 5: ANÁLISIS CON IA ---
    st.markdown("---")
    st.header("5. Análisis Experto con Gemini AI")

    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # ¡CORRECCIÓN 4: Cambiado a un modelo válido!
            model = genai.GenerativeModel('gemini-2.5-flash')

            if st.button("Generar Resumen"):
                # PREPARACIÓN DE DATOS PARA LA IA
                resumen_stats = df[columna].describe().to_string()
                
                prompt_ingenieria = f"""
                Actúa como un experto en Ciencia de Datos y Estadística. 
                Analiza los siguientes resultados obtenidos de la variable '{columna}':
                
                ESTADÍSTICOS:
                {resumen_stats}
                
                PRUEBA DE NORMALIDAD (Shapiro-Wilk):
                - P-Value: {p_value:.4f}
                - Veredicto: {'Normal' if p_value > 0.05 else 'No Normal'}
                
                TAREA: 
                Dime que se observa de manera didactica y sencilla, que no sea tan extenso
                si exposible de un parrafo, de manera que se entienda, ya que no soy experto
                en estadística, pero quiero entender qué significan estos resultados y cómo 
                podrían afectar la interpretación de mis datos, que no sea nada extenso y que 
                contenga emojis.
                """
                
                with st.spinner("Gemini está redactando el análisis..."):
                    response = model.generate_content(prompt_ingenieria)
                    st.info("### Conclusión Técnica Sugerida")
                    st.write(response.text)
                    
        except Exception as e:
            st.error(f"Error de conexión con la API: {e}")
    else:
        st.warning("⚠️ Configura tu API Key en el archivo .env para habilitar este módulo.")