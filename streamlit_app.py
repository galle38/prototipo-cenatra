import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import plotly.express as px


st.set_page_config(page_title="Prototipo CENATRA", page_icon="馃珋", layout="wide")

DATA_PATH = Path("data/trasplantes.csv")
MODEL_RLM_PATH = Path("modelo_rlm.pkl")
MODEL_OLS_PATH = Path("modelo_ols.pkl")

REQUIRED_COLUMNS = [
    "organo",
    "sexo",
    "edad_al_trasplante_anios",
    "grupo_sanguineo_receptor",
    "rh",
    "relacion",
    "institucion",
    "tipo_trasplante",
    "entidad_federativa_trasplante",
    "tiempo_espera_dias",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "fecha_trasplante" in df.columns:
        df["fecha_trasplante"] = pd.to_datetime(df["fecha_trasplante"], errors="coerce")
    if "fecha_registro_comite" in df.columns:
        df["fecha_registro_comite"] = pd.to_datetime(df["fecha_registro_comite"], errors="coerce")
    if "tiempo_espera_dias" not in df.columns and {"fecha_trasplante", "fecha_registro_comite"}.issubset(df.columns):
        df["tiempo_espera_dias"] = (df["fecha_trasplante"] - df["fecha_registro_comite"]).dt.days
    return df


@st.cache_resource
def load_models():
    with open(MODEL_RLM_PATH, "rb") as f:
        model_rlm = pickle.load(f)
    with open(MODEL_OLS_PATH, "rb") as f:
        model_ols = pickle.load(f)
    return model_rlm, model_ols


def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en el dataset: {missing}")
        st.stop()

    out = df.copy()
    out = out[out["edad_al_trasplante_anios"].notna()]
    out = out[out["tiempo_espera_dias"].notna()]
    out = out[out["tiempo_espera_dias"] >= 0]
    return out


def build_input_row(df: pd.DataFrame) -> pd.DataFrame:
    col1, col2, col3 = st.columns(3)

    with col1:
        organo = st.selectbox("脫rgano", sorted(df["organo"].dropna().astype(str).unique()))
        sexo = st.selectbox("Sexo", sorted(df["sexo"].dropna().astype(str).unique()))
        edad = st.number_input("Edad", min_value=0, max_value=100, value=45, step=1)

    with col2:
        grupo = st.selectbox(
            "Grupo sangu铆neo",
            sorted(df["grupo_sanguineo_receptor"].dropna().astype(str).unique()),
        )
        rh = st.selectbox("RH", sorted(df["rh"].dropna().astype(str).unique()))
        relacion = st.selectbox("Relaci贸n con el donante", sorted(df["relacion"].dropna().astype(str).unique()))

    with col3:
        institucion = st.selectbox("Instituci贸n", sorted(df["institucion"].dropna().astype(str).unique()))
        tipo_trasplante = st.selectbox(
            "Tipo de trasplante",
            sorted(df["tipo_trasplante"].dropna().astype(str).unique()),
        )
        entidad = st.selectbox(
            "Entidad federativa del trasplante",
            sorted(df["entidad_federativa_trasplante"].dropna().astype(str).unique()),
        )

    row = pd.DataFrame(
        [
            {
                "organo": organo,
                "sexo": sexo,
                "edad_al_trasplante_anios": edad,
                "grupo_sanguineo_receptor": grupo,
                "rh": rh,
                "relacion": relacion,
                "institucion": institucion,
                "tipo_trasplante": tipo_trasplante,
                "entidad_federativa_trasplante": entidad,
            }
        ]
    )
    return row


def render_summary(df: pd.DataFrame):
    st.subheader("Resumen del dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df):,}")
    c2.metric("Tiempo medio de espera", f"{df['tiempo_espera_dias'].mean():.1f} d铆as")
    c3.metric("Mediana de espera", f"{df['tiempo_espera_dias'].median():.1f} d铆as")
    c4.metric("Edad media", f"{df['edad_al_trasplante_anios'].mean():.1f} a帽os")


def render_charts(df: pd.DataFrame):
    st.subheader("Visualizaciones")

    chart_col1, chart_col2 = st.columns(2)
    col1, col2 = st.columns([1, 1])  # mismo tama帽o


    with chart_col1:
      
        import plotly.express as px
        # Quitar valores inv谩lidos para log (0 o negativos)
        df_clean = df[df["tiempo_espera_dias"] > 0]
        
        # Histograma
        fig = px.histogram(
        df_clean,
        x="tiempo_espera_dias",
        nbins=30,
        title="Distribuci贸n del tiempo de espera",
        labels={
            "tiempo_espera_dias": "D铆as",
            "count": "Frecuencia"
        }
        )
        
        fig.update_layout(
            yaxis_title="Frecuencia"
        )
       
            # Aplicar escala logar铆tmica
       # fig.update_xaxes(type="log")
        # Mostrar en Streamlit
        st.plotly_chart(fig, use_container_width=True)
                  

    with chart_col2:
        #----------------Conteo total de 贸rganos---------------------
        import plotly.express as px
        
        conteo = df["organo"].value_counts().reset_index()
        conteo.columns = ["organo", "conteo"]
        fig = px.bar(
            conteo,
            x="organo",
            y="conteo",
            color="organo",  # 馃憟 esto da un color distinto a cada barra
            #text="conteo"
        )
        
        # Tooltip personalizado
        fig.update_traces(
            hovertemplate="脫rgano: %{x}<br>Trasplantes: %{y}<extra></extra>"
        )
        
        # Est茅tica
        fig.update_layout(
            title="Conteo de 贸rganos trasplantados",
            xaxis_title="脫rgano",
            yaxis_title="Frecuencia",
            showlegend=False  # opcional, porque ya se ve en el eje X
        )
            # Aplicar escala logar铆tmica
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        ##------------------------------------------------

#--------------------------Promedio de tiempo de espera por 贸rgano con tooltip--------------------------------------------------------------------------
    import plotly.express as px
    
    avg_wait = df.groupby("organo", dropna=False)["tiempo_espera_dias"].mean().sort_values(ascending=False).head(10)   
    fig = px.bar(
        x=avg_wait.index.astype(str),
        y=avg_wait.values,
        color=avg_wait.index.astype(str),  # 馃憟 esto asigna un color por 贸rgano
        labels={"x": "脫rgano", "y": "D铆as"},
        title="Promedio de tiempo de espera por 贸rgano"
    )
    
    # Personalizar tooltip
    fig.update_traces(
        hovertemplate="<b>脫rgano:</b> %{x}<br><b>Promedio:</b> %{y:.2f} d铆as"
    )
    
    st.plotly_chart(fig)
#---------------------------------------------------------------------------------------------------------------------


def interpret_prediction(pred: float, df: pd.DataFrame):
    p25 = df["tiempo_espera_dias"].quantile(0.25)
    p50 = df["tiempo_espera_dias"].quantile(0.50)
    p75 = df["tiempo_espera_dias"].quantile(0.75)

    if pred <= p25:
        st.success("Estimaci贸n relativamente baja respecto a la distribuci贸n hist贸rica.")
    elif pred <= p50:
        st.info("Estimaci贸n cercana al rango bajo-medio de la distribuci贸n hist贸rica.")
    elif pred <= p75:
        st.warning("Estimaci贸n por encima de la mediana hist贸rica.")
    else:
        st.error("Estimaci贸n alta respecto a la distribuci贸n hist贸rica observada.")

    st.caption(
        f"Referencia hist贸rica: Q1={p25:.1f} d铆as, mediana={p50:.1f} d铆as, Q3={p75:.1f} d铆as."
    )


def render_prediction(model_rlm, model_ols, input_row: pd.DataFrame, df: pd.DataFrame):
    st.subheader("Predicci贸n")

    pred_rlm = max(float(model_rlm.predict(input_row).iloc[0]), 0.0)
    pred_ols = max(float(model_ols.predict(input_row).iloc[0]), 0.0)
    diferencia = pred_ols - pred_rlm

    c1, c2, c3 = st.columns(3)
    c1.metric("Tiempo estimado RLM", f"{pred_rlm:.1f} d铆as")
    c2.metric("Tiempo estimado OLS", f"{pred_ols:.1f} d铆as")
    c3.metric("Diferencia OLS - RLM", f"{diferencia:.1f} d铆as")

    st.write("**Interpretaci贸n con RLM**")
    interpret_prediction(pred_rlm, df)

    st.write("**Perfil evaluado**")
    st.dataframe(input_row, use_container_width=True)


def compute_comparison(df: pd.DataFrame, model_rlm, model_ols):
    y_true = df["tiempo_espera_dias"]
    y_pred_rlm = np.maximum(model_rlm.predict(df), 0)
    y_pred_ols = np.maximum(model_ols.predict(df), 0)

    coef_comparacion = pd.DataFrame({
        "OLS": model_ols.params,
        "RLM_HuberT": model_rlm.params,
    }).fillna(0)
    coef_comparacion["Diferencia_absoluta"] = (coef_comparacion["OLS"] - coef_comparacion["RLM_HuberT"]).abs()

    metricas = pd.DataFrame({
        "Modelo": ["OLS", "RLM_HuberT"],
        "MAE": [
            mean_absolute_error(y_true, y_pred_ols),
            mean_absolute_error(y_true, y_pred_rlm),
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_true, y_pred_ols)),
            np.sqrt(mean_squared_error(y_true, y_pred_rlm)),
        ],
        "Error_medio": [
            np.mean(y_true - y_pred_ols),
            np.mean(y_true - y_pred_rlm),
        ],
    })
    return coef_comparacion, metricas, y_true, y_pred_ols, y_pred_rlm


def render_comparison_tab(df: pd.DataFrame, model_rlm, model_ols):
    st.subheader("Comparaci贸n de modelos")
    coef_comparacion, metricas, y_true, y_pred_ols, y_pred_rlm = compute_comparison(df, model_rlm, model_ols)

    m1, m2 = st.columns(2)
    best_mae = metricas.loc[metricas["MAE"].idxmin(), "Modelo"]
    best_rmse = metricas.loc[metricas["RMSE"].idxmin(), "Modelo"]
    m1.metric("Mejor MAE", best_mae)
    m2.metric("Mejor RMSE", best_rmse)

    st.write("**M茅tricas de error**")
    st.dataframe(metricas.round(3), use_container_width=True)

    st.write("**Comparaci贸n de coeficientes**")
    st.dataframe(coef_comparacion.round(4), use_container_width=True)
    
#------------------------------Comparaci贸n de residuos: OLS vs RLM-HuberT---------------------------------------------------------
    import plotly.graph_objects as go
    #import streamlit as st
    
    # Crear figura
    fig = go.Figure()
    
    # Scatter OLS
    fig.add_trace(
        go.Scatter(
            x=y_pred_ols,
            y=y_true - y_pred_ols,
            mode='markers',
            name='OLS',
            opacity=0.5
        )
    )
    
    # Scatter RLM
    fig.add_trace(
        go.Scatter(
            x=y_pred_rlm,
            y=y_true - y_pred_rlm,
            mode='markers',
            name='RLM HuberT',
            opacity=0.5
        )
    )
    
    # L铆nea horizontal en 0
    fig.add_hline(
        y=0,
        line_dash="dash"
    )
    
    # Layout
    fig.update_layout(
        title="Comparaci贸n de residuos: OLS vs RLM-HuberT",
        xaxis_title="Predicci贸n",
        yaxis_title="Residuo",
        template="plotly_white"
    )
    
    # Mostrar en Streamlit
    st.plotly_chart(fig, use_container_width=True)

#---------------------------------------------------------------------------------------
    col1, col2 = st.columns(2)

#------Distribuci贸n de residuo------------------------------------------------------------------------
    with col1:
        #st.plotly_chart(fig_coeficientes, use_container_width=True)  
        residuos_ols = y_true - y_pred_ols
        residuos_rlm = y_true - y_pred_rlm
    
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=residuos_ols,
            nbinsx=30,
            opacity=0.6,
            name="OLS"
        ))
        
        fig.add_trace(go.Histogram(
            x=residuos_rlm,
            nbinsx=30,
            opacity=0.6,
            name="RLM HuberT"
        ))
        
        fig.update_layout(
            height=600,
            title="Distribuci贸n de residuos",
            xaxis_title="Residuo",
            yaxis_title="Frecuencia",
            barmode="overlay"  # para que se sobrepongan como en matplotlib
        )

        st.plotly_chart(fig, use_container_width=True)
        #st.plotly_chart(fig_coeficientes, use_container_width=True)  
#--------Top coeficientes con mayor diferencia absoluta------------------------------------------------------------------------
    with col2:
      
        import plotly.express as px
        
        # Obtener top 15 diferencias
        top_diff = coef_comparacion["Diferencia_absoluta"] \
            .sort_values(ascending=False) \
            .head(15)
        
        # Crear DataFrame para Plotly
        df_plot = top_diff.reset_index()
        df_plot.columns = ["Variable", "Diferencia_absoluta"]
        
        # Gr谩fica interactiva 
        fig = px.bar(
            df_plot,
            x="Variable",
            y="Diferencia_absoluta",
            color="Diferencia_absoluta",
            color_continuous_scale="viridis",
            title="Top coeficientes con mayor diferencia absoluta",
            height=600
        )
        
        # Rotar etiquetas del eje X
        fig.update_layout(
            xaxis_tickangle=-90
        )
        
        # Tooltip personalizado
        fig.update_traces(
            hovertemplate="<b>Variable:</b> %{x}<br><b>Diferencia:</b> %{y:.4f}"
        )
        
        # Mostrar en Streamlit
        st.plotly_chart(fig, use_container_width=True)
    
#--------------------------------------------------------------------------------
    st.write(
        "Un menor MAE o RMSE indica mejor capacidad predictiva promedio. Si RLM presenta errores menores o residuos m谩s estables, "
        "ello respalda su uso en presencia de asimetr铆a y valores at铆picos."
    )


def main():
    st.title("馃珋 Prototipo anal铆tico CENATRA")
    st.write(
        "Aplicaci贸n demostrativa para estimar el tiempo de espera de un trasplante a partir de variables demogr谩ficas y administrativas."
    )

    try:
        df = validate_df(load_data())
        model_rlm, model_ols = load_models()
    except FileNotFoundError as e:
        st.error(f"No se encontr贸 un archivo requerido: {e}")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Simulador", "Dashboard", "Comparaci贸n", "Datos"])

    with tab1:
        st.write("Complete el formulario con un perfil compatible con las categor铆as observadas en el dataset.")
        input_row = build_input_row(df)
        if st.button("Predecir tiempo de espera"):
                        render_prediction(model_rlm, model_ols, input_row, df)

    with tab2:
        render_summary(df)
        render_charts(df)

    with tab3:
        render_comparison_tab(df, model_rlm, model_ols)

    with tab4:
        st.subheader("Vista previa del dataset")
        st.dataframe(df.head(50), use_container_width=True)
        st.download_button(
            "Descargar muestra CSV",
            data=df.head(200).to_csv(index=False).encode("utf-8"),
            file_name="muestra_trasplantes.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
