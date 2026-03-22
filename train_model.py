import pickle
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Cargar dataset
df = pd.read_csv("data/trasplantes_original.csv")

# Fechas
df["fecha_trasplante"] = pd.to_datetime(df["fecha_trasplante"], errors="coerce")
df["fecha_registro_comite"] = pd.to_datetime(df["fecha_registro_comite"], errors="coerce")

# Variable objetivo
df["tiempo_espera_dias"] = (
    df["fecha_trasplante"] - df["fecha_registro_comite"]
).dt.days

# Variables del modelo
columnas = [
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

df = df[columnas].copy()
df = df.dropna()
df = df[df["tiempo_espera_dias"] >= 0]
df = df[df["edad_al_trasplante_anios"] >= 0]

print("Datos listos:", df.shape)

formula = """
tiempo_espera_dias ~ C(organo) + C(sexo) + edad_al_trasplante_anios
+ C(grupo_sanguineo_receptor) + C(rh) + C(relacion)
+ C(institucion) + C(tipo_trasplante) + C(entidad_federativa_trasplante)
"""

# Modelo OLS
modelo_ols = smf.ols(formula=formula, data=df).fit()

# Modelo RLM
modelo_rlm = smf.rlm(
    formula=formula,
    data=df,
    M=sm.robust.norms.HuberT()
).fit()

# Guardar modelos
with open("modelo_ols.pkl", "wb") as f:
    pickle.dump(modelo_ols, f)

with open("modelo_rlm.pkl", "wb") as f:
    pickle.dump(modelo_rlm, f)

print("Modelos guardados: modelo_ols.pkl y modelo_rlm.pkl")

# Guardar dataset limpio que usará la app
df.to_csv("data/trasplantes.csv", index=False)


# Comparación rápida en consola
y_true = df["tiempo_espera_dias"]
y_pred_ols = np.maximum(modelo_ols.predict(df), 0)
y_pred_rlm = np.maximum(modelo_rlm.predict(df), 0)

coef_comparacion = pd.DataFrame({
    "OLS": modelo_ols.params,
    "RLM_HuberT": modelo_rlm.params
}).fillna(0)

metricas = pd.DataFrame({
    "Modelo": ["OLS", "RLM_HuberT"],
    "MAE": [
        mean_absolute_error(y_true, y_pred_ols),
        mean_absolute_error(y_true, y_pred_rlm)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_true, y_pred_ols)),
        np.sqrt(mean_squared_error(y_true, y_pred_rlm))
    ]
})

print("\n=== Comparación de coeficientes ===")
print(coef_comparacion.head(20))

print("\n=== Comparación de errores ===")
print(metricas)

print("\n=== Summary OLS ===")
print(modelo_ols.summary())

print("\n=== Summary RLM ===")
print(modelo_rlm.summary())