import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
from dotenv import load_dotenv
import google.generativeai as genai


def plot_z_distribution(z_stat, alpha, tipo_cola):
    # Generar puntos para la curva normal estándar
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, color='black', label='Normal Estándar (H0)')
    
    # Colorear zonas de rechazo según el tipo de prueba
    if tipo_cola == "Bilateral (≠)":
        z_critico = stats.norm.ppf(1 - alpha/2)
        ax.fill_between(x, 0, y, where=(x > z_critico) | (x < -z_critico), color='red', alpha=0.3, label='Zona Rechazo')
    elif tipo_cola == "Cola Derecha (>)":
        z_critico = stats.norm.ppf(1 - alpha)
        ax.fill_between(x, 0, y, where=(x > z_critico), color='red', alpha=0.3, label='Zona Rechazo')
    else: # Cola Izquierda
        z_critico = stats.norm.ppf(alpha)
        ax.fill_between(x, 0, y, where=(x < z_critico), color='red', alpha=0.3, label='Zona Rechazo')
        
    # Dibujar línea del estadístico calculado
    ax.axvline(z_stat, color='blue', linestyle='--', lw=2, label=f'Z calculado: {z_stat:.2f}')
    ax.legend(fontsize='small')
    return fig

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
if df is not None:
    # Cálculos Estadísticos
    datos = df[columna].dropna()
    n_size, media_m, desv_p = len(datos), datos.mean(), datos.std()
    z_stat = (media_m - h0_valor) / (desv_p / np.sqrt(n_size))

    # --- CÁLCULOS DE DIAGNÓSTICO ---
    # 1. Sesgo (Skewness)
    valor_sesgo = datos.skew()
    if valor_sesgo > 0.5:
        interpretacion_sesgo = "Sesgo positivo (derecha) ➡️"
    elif valor_sesgo < -0.5:
        interpretacion_sesgo = "Sesgo negativo (izquierda) ⬅️"
    else:
        interpretacion_sesgo = "Distribución simétrica ✅"

    # 2. Outliers (IQR)
    Q1 = datos.quantile(0.25)
    Q3 = datos.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = datos[(datos < limite_inferior) | (datos > limite_superior)]
    total_outliers = len(outliers)

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

        fig_z = plot_z_distribution(z_stat, alpha, tipo_cola)
        st.pyplot(fig_z)

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

    with st.expander("🔍 Ver Análisis Detallado de la Forma"):
        col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Simetría y Sesgo**")
        st.metric("Skewness", f"{valor_sesgo:.2f}")
        st.info(f"Estado: {interpretacion_sesgo}")
        
    with col2:
        st.write("**Valores Atípicos (Outliers)**")
        st.metric("Detectados", f"{total_outliers}")
        if total_outliers > 0:
            st.warning(f"Se encontraron {total_outliers} valores fuera de los límites esperados.")
        else:
            st.success("No se detectaron outliers significativos.")

# --- MÓDULO DE IA (UNIFICADO) ---
    if api_key:
        st.markdown("---")
        st.subheader("🤖 Validación con Inteligencia Artificial")

        # 1. El estudiante elige primero
        decision_estudiante = st.radio(
            "Antes de consultar a la IA, ¿cuál es tu conclusión?",
            ["Rechazar H0", "No Rechazar H0"],
            index=None
        )

        # 2. Un solo botón que hace todo
        if decision_estudiante:
            if st.button("🚀 Obtener Veredicto de Gemini"):
                with st.spinner("La IA está analizando tus resultados..."):
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        # Resumen técnico solicitado por el profe
                        resumen_stats = f"""
                        PRUEBA Z:
                        - Media muestral={media_m:.4f}, H0={h0_valor:.4f}, n={n_size}
                        - Z={z_stat:.4f}, P-value={p_value_z:.4f}, Alpha={alpha}
                        - Diagnóstico: Sesgo={valor_sesgo:.2f}, Outliers={total_outliers}
                        - Decisión Estudiante: {decision_estudiante}
                        """
                        
                        prompt = f"""
                        Analiza estos resultados de una prueba Z: {resumen_stats}
                        
                        Responde de forma técnica y MUY breve (máximo 80 palabras):
                        1. DECISIÓN: ¿Es correcta la decisión del estudiante basándose en P-value vs Alpha?
                        2. SUPUESTOS: ¿El sesgo o los outliers comprometen la prueba?
                        3. INFERENCIA: Una frase final sobre la variable {columna}.
                        """
                        
                        response = model.generate_content(prompt)
                        st.markdown("---")
                        st.info(response.text)
                        
                        # Validación visual rápida
                        es_correcto = ("Rechazar" in decision_estudiante and p_value_z < alpha) or \
                                      ("No Rechazar" in decision_estudiante and p_value_z >= alpha)
                        
                        if es_correcto:
                            st.success("✨ ¡Tu conclusión coincide con la lógica estadística!")
                        else:
                            st.warning("⚠️ Tu conclusión difiere del criterio del P-value. ¡Revisa la teoría!")

                    except Exception as e:
                        st.error(f"Error técnico: {e}")

else:
    # --- PANTALLA DE BIENVENIDA OPTIMIZADA (VISTA ÚNICA) ---
    
    # 1. Encabezado con Identidad de Ingeniería
    st.markdown("""
        <div style="background-color: #003366; padding: 40px; border-radius: 20px; text-align: center; margin-bottom: 30px; border: 2px solid #00c4cc;">
            <h1 style="font-size: 3.5rem; color: #ffffff; margin-bottom: 10px; font-family: 'Helvetica', sans-serif;">📊 Analizador Estadístico Inteligente</h1>
            <p style="font-size: 1.4rem; color: #00c4cc; font-weight: bold;">Ingeniería en Tecnologías de Información e Innovación Digital</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. Instrucción en una sola línea para ahorrar espacio
    st.info("💡 **Configuración:** Usa el menú lateral izquierdo para cargar datos y configurar la prueba.")

    # 3. Mosaico de Capacidades (Sin saltos de línea innecesarios)
    col_stat, col_diag, col_gemini = st.columns(3)
    
    with col_stat:
        st.markdown("### 📉 **Inferencia Z**")
        st.markdown("<small>Cálculo de estadísticos, P-value y veredictos bajo estándares rigurosos.</small>", unsafe_allow_html=True)
        st.caption("Soporta pruebas bilaterales y de cola única.")

    with col_diag:
        st.markdown("### 🔍 **Diagnóstico**")
        st.markdown("<small>Análisis automático de Skewness (sesgo) y detección de Outliers (IQR).</small>", unsafe_allow_html=True)
        st.caption("Validación de supuestos de normalidad.")

    with col_gemini:
        st.markdown("### 🤖 **Veredicto IA**")
        st.markdown("<small>Integración con Gemini para validar la consistencia de tus conclusiones.</small>", unsafe_allow_html=True)
        st.caption("Consultoría con modelos generativos.")

    # 4. Pie de Página Institucional Minimalista
    st.markdown("---")
    st.markdown("""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 10px; text-align: center; border-left: 5px solid #003366;">
            <p style="margin: 0; color: #003366; font-size: 0.9rem; font-weight: bold;">🎓 Universidad Politécnica de Chiapas</p>
            <p style="margin: 0; color: #555; font-size: 0.8rem;">Estudiante: Elizabeth Escobar | Ciclo Escolar 2026</p>
        </div>
    """, unsafe_allow_html=True)