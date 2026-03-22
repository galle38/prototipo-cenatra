import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Prototipo CENATRA", page_icon="🫀", layout="wide")

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
        organo = st.selectbox("Órgano", sorted(df["organo"].dropna().astype(str).unique()))
        sexo = st.selectbox("Sexo", sorted(df["sexo"].dropna().astype(str).unique()))
        edad = st.number_input("Edad", min_value=0, max_value=100, value=45, step=1)

    with col2:
        grupo = st.selectbox(
            "Grupo sanguíneo",
            sorted(df["grupo_sanguineo_receptor"].dropna().astype(str).unique()),
        )
        rh = st.selectbox("RH", sorted(df["rh"].dropna().astype(str).unique()))
        relacion = st.selectbox("Relación con el donante", sorted(df["relacion"].dropna().astype(str).unique()))

    with col3:
        institucion = st.selectbox("Institución", sorted(df["institucion"].dropna().astype(str).unique()))
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
    c2.metric("Tiempo medio de espera", f"{df['tiempo_espera_dias'].mean():.1f} días")
    c3.metric("Mediana de espera", f"{df['tiempo_espera_dias'].median():.1f} días")
    c4.metric("Edad media", f"{df['edad_al_trasplante_anios'].mean():.1f} años")


def render_charts(df: pd.DataFrame):
    st.subheader("Visualizaciones")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = plt.figure()
        plt.hist(df["tiempo_espera_dias"].dropna(), bins=30)
        plt.title("Distribución del tiempo de espera")
        plt.xlabel("Días")
        plt.ylabel("Frecuencia")
        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        top_org = df["organo"].value_counts().head(5)
        fig = plt.figure()
        plt.bar(top_org.index.astype(str), top_org.values)
        plt.title("Top 5 órganos trasplantados")
        plt.xticks(rotation=45)
        plt.ylabel("Frecuencia")
        st.pyplot(fig)
        plt.close(fig)

    avg_wait = df.groupby("organo", dropna=False)["tiempo_espera_dias"].mean().sort_values(ascending=False).head(10)
    fig = plt.figure()
    plt.bar(avg_wait.index.astype(str), avg_wait.values)
    plt.title("Promedio de tiempo de espera por órgano")
    plt.xticks(rotation=45)
    plt.ylabel("Días")
    st.pyplot(fig)
    plt.close(fig)


def interpret_prediction(pred: float, df: pd.DataFrame):
    p25 = df["tiempo_espera_dias"].quantile(0.25)
    p50 = df["tiempo_espera_dias"].quantile(0.50)
    p75 = df["tiempo_espera_dias"].quantile(0.75)

    if pred <= p25:
        st.success("Estimación relativamente baja respecto a la distribución histórica.")
    elif pred <= p50:
        st.info("Estimación cercana al rango bajo-medio de la distribución histórica.")
    elif pred <= p75:
        st.warning("Estimación por encima de la mediana histórica.")
    else:
        st.error("Estimación alta respecto a la distribución histórica observada.")

    st.caption(
        f"Referencia histórica: Q1={p25:.1f} días, mediana={p50:.1f} días, Q3={p75:.1f} días."
    )


def render_prediction(model_rlm, model_ols, input_row: pd.DataFrame, df: pd.DataFrame):
    st.subheader("Predicción")

    pred_rlm = max(float(model_rlm.predict(input_row).iloc[0]), 0.0)
    pred_ols = max(float(model_ols.predict(input_row).iloc[0]), 0.0)
    diferencia = pred_ols - pred_rlm

    c1, c2, c3 = st.columns(3)
    c1.metric("Tiempo estimado RLM", f"{pred_rlm:.1f} días")
    c2.metric("Tiempo estimado OLS", f"{pred_ols:.1f} días")
    c3.metric("Diferencia OLS - RLM", f"{diferencia:.1f} días")

    st.write("**Interpretación con RLM**")
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
    st.subheader("Comparación de modelos")
    coef_comparacion, metricas, y_true, y_pred_ols, y_pred_rlm = compute_comparison(df, model_rlm, model_ols)

    m1, m2 = st.columns(2)
    best_mae = metricas.loc[metricas["MAE"].idxmin(), "Modelo"]
    best_rmse = metricas.loc[metricas["RMSE"].idxmin(), "Modelo"]
    m1.metric("Mejor MAE", best_mae)
    m2.metric("Mejor RMSE", best_rmse)

    st.write("**Métricas de error**")
    st.dataframe(metricas.round(3), use_container_width=True)

    st.write("**Comparación de coeficientes**")
    st.dataframe(coef_comparacion.round(4), use_container_width=True)

    fig = plt.figure()
    plt.scatter(y_pred_ols, y_true - y_pred_ols, alpha=0.5, label="OLS")
    plt.scatter(y_pred_rlm, y_true - y_pred_rlm, alpha=0.5, label="RLM HuberT")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicción")
    plt.ylabel("Residuo")
    plt.title("Comparación de residuos: OLS vs RLM-HuberT")
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)

    residuos_ols = y_true - y_pred_ols
    residuos_rlm = y_true - y_pred_rlm

    fig = plt.figure()
    plt.hist(residuos_ols, bins=30, alpha=0.6, label="OLS")
    plt.hist(residuos_rlm, bins=30, alpha=0.6, label="RLM HuberT")
    plt.title("Distribución de residuos")
    plt.xlabel("Residuo")
    plt.ylabel("Frecuencia")
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)

    fig = plt.figure()
    top_diff = coef_comparacion["Diferencia_absoluta"].sort_values(ascending=False).head(15)
    plt.bar(top_diff.index.astype(str), top_diff.values)
    plt.title("Top coeficientes con mayor diferencia absoluta")
    plt.xticks(rotation=90)
    plt.ylabel("Diferencia absoluta")
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "Un menor MAE o RMSE indica mejor capacidad predictiva promedio. Si RLM presenta errores menores o residuos más estables, "
        "ello respalda su uso en presencia de asimetría y valores atípicos."
    )


def main():
    st.title("🫀 Prototipo analítico CENATRA")
    st.write(
        "Aplicación demostrativa para estimar el tiempo de espera de un trasplante a partir de variables demográficas y administrativas."
    )

    try:
        df = validate_df(load_data())
        model_rlm, model_ols = load_models()
    except FileNotFoundError as e:
        st.error(f"No se encontró un archivo requerido: {e}")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Simulador", "Dashboard", "Comparación", "Datos"])

    with tab1:
        st.write("Complete el formulario con un perfil compatible con las categorías observadas en el dataset.")
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
