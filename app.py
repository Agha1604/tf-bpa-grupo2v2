import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

st.set_page_config(page_title="Intención de compra - Grupo 2", page_icon="🛒", layout="wide")

cols_num = [
    "administrative", "administrative_duration",
    "informational", "informational_duration",
    "product_related", "product_related_duration",
    "page_values", "special_day"
]
cols_nominales = ["month", "visitor_type"]
cols_bool = ["weekend"]
cols_modelo = cols_num + cols_nominales + cols_bool
orden_meses = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# Misma limpieza que el notebook
def limpiar(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        "productrelated": "product_related",
        "productrelated_duration": "product_related_duration",
        "bouncerates": "bounce_rates",
        "exitrates": "exit_rates",
        "pagevalues": "page_values",
        "specialday": "special_day",
        "operatingsystems": "operating_systems",
        "traffictype": "traffic_type",
        "visitortype": "visitor_type",
    })
    df = df.astype({"month": "string", "visitor_type": "string"})
    df = df.drop_duplicates()
    df = df.drop(columns=["operating_systems", "browser", "region",
                          "bounce_rates", "exit_rates", "traffic_type"])
    df["venta"] = df["revenue"].astype(str).str.lower().eq("true").astype("int8")
    return df


def a_entero(x):
    return x.astype(int)


def crear_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]), cols_num),
        ("cat_nom", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))
        ]), cols_nominales),
        ("bool", Pipeline([
            ("to_int", FunctionTransformer(a_entero, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="if_binary", sparse_output=False))
        ]), cols_bool)
    ])


def crear_xgb():
    # parámetros que eligió el GridSearch del notebook
    return XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8,
                         random_state=42, eval_metric="logloss", n_jobs=-1)


def metricas(nombre, y_real, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, pred).ravel()
    return {
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_real, pred),
        "Precisión (compra)": precision_score(y_real, pred, zero_division=0),
        "Recall (compra)": recall_score(y_real, pred),
        "F1 (compra)": f1_score(y_real, pred),
        "ROC-AUC": roc_auc_score(y_real, prob),
        "PR-AUC": average_precision_score(y_real, prob),
        "Falsos positivos": fp,
        "Falsos negativos": fn,
    }


# ---------------- barra lateral ----------------
st.sidebar.title("🧭 Navegación")
seccion = st.sidebar.radio("Selecciona una sección:", [
    "📊 Exploración de Datos",
    "⚙️ Entrenamiento del Modelo",
    "🔮 Hacer Predicciones",
    "📈 Análisis de Resultados",
])
st.sidebar.markdown("---")
st.sidebar.caption("Business Predictive Analytics - UPC\n\nGrupo 2 - Sección 9341")

st.title("🛒 Predicción de Intención de Compra en E-Commerce")
st.markdown("### Aplicación interactiva con sesiones de navegación web")
st.markdown("---")

# ---------------- 1. exploración ----------------
if seccion == "📊 Exploración de Datos":
    st.header("📊 Exploración y Análisis del Dataset")

    archivo = st.file_uploader("Sube el archivo online_shoppers_intention.csv:", type="csv")
    usar_local = st.button("Usar el dataset del proyecto (UCI)")

    if archivo is not None:
        st.session_state["df"] = limpiar(pd.read_csv(archivo))
    elif usar_local:
        st.session_state["df"] = limpiar(pd.read_csv("online_shoppers_intention.csv"))

    if "df" not in st.session_state:
        st.info("👆 Sube el archivo CSV o usa el dataset del proyecto para comenzar.")
        st.markdown("""
        El archivo debe tener las columnas originales del dataset de UCI (Sakar et al., 2019), entre ellas:
        - `Administrative`, `Informational`, `ProductRelated` y sus duraciones
        - `PageValues`, `SpecialDay`, `Month`, `VisitorType`, `Weekend`
        - `Revenue`: variable objetivo (True = la sesión terminó en compra)
        """)
    else:
        df = st.session_state["df"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sesiones (sin duplicados)", f"{len(df):,}".replace(",", " "))
        c2.metric("Variables del modelo", len(cols_modelo))
        c3.metric("Valores nulos", int(df.isnull().sum().sum()))
        c4.metric("Tasa de compra", f"{df['venta'].mean() * 100:.1f}%")

        st.markdown("**Supuestos:** cada sesión es independiente (no hay ID de usuario) y cada mes "
                    "es un ciclo de compra independiente, por eso los datos se separan al azar.")

        st.subheader("Primeras filas")
        st.dataframe(df[cols_modelo + ["revenue"]].head(10))

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Distribución del target")
            conteo = df["revenue"].astype(str).value_counts()
            st.bar_chart(conteo, y_label="Sesiones", x_label="revenue")
            st.caption(f"Solo el {df['venta'].mean() * 100:.1f}% de las sesiones compra: datos desbalanceados.")
        with col_b:
            st.subheader("Conversión por mes (%)")
            meses = (df.groupby("month")["venta"].mean() * 100).reindex(orden_meses).round(2)
            st.bar_chart(meses, y_label="Conversión (%)", x_label="Mes")
            st.caption("Noviembre convierte mucho más que febrero.")

        st.subheader("Conversión por tiempo en páginas de producto")
        rangos = pd.cut(df["product_related_duration"] / 60, bins=[-1, 2, 10, 30, 1000],
                        labels=["0 - 2 min", "2 - 10 min", "10 - 30 min", "+30 min"])
        tabla = df.groupby(rangos, observed=True)["venta"].agg(sesiones="size", compras="sum")
        tabla["conversión (%)"] = (tabla["compras"] / tabla["sesiones"] * 100).round(2)
        st.dataframe(tabla)
        st.success("✅ Datos cargados. Ahora ve a **Entrenamiento del Modelo**.")

# ---------------- 2. entrenamiento ----------------
elif seccion == "⚙️ Entrenamiento del Modelo":
    st.header("⚙️ Entrenamiento del Modelo")

    if "df" not in st.session_state:
        st.warning("⚠️ Por favor, carga primero el dataset en la sección de Exploración de Datos.")
    else:
        df = st.session_state["df"]
        st.markdown("Modelo: **XGBoost** con los parámetros del GridSearch del notebook "
                    "(`n_estimators=100`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`).")

        c1, c2 = st.columns(2)
        with c1:
            test_size = st.slider("Porcentaje para test", 0.10, 0.40, 0.20, 0.05)
            tecnica = st.selectbox("Técnica de balanceo",
                                   ["Sin balanceo", "Random Oversampling", "SMOTE", "Random Undersampling"])
        with c2:
            sampling = st.slider("sampling_strategy (compras / no compras en train)", 0.2, 1.0, 0.5, 0.1,
                                 disabled=(tecnica == "Sin balanceo"))
            vecinos = st.slider("Vecinos de SMOTE (k_neighbors)", 1, 10, 5, disabled=(tecnica != "SMOTE"))

        st.caption("Con 15.6% de compras el balanceo no es obligatorio. En el notebook lo dejamos sin balanceo "
                   "porque subía los falsos positivos (ventas perdidas).")

        if st.button("🚀 Entrenar modelo", type="primary"):
            X = df.drop(columns=["venta", "revenue"])
            y = df["venta"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y)

            # winsorización con el P98 de train
            tope_alto = round(np.percentile(X_train["product_related_duration"], 98), 2)
            X_train["product_related_duration"] = X_train["product_related_duration"].clip(upper=tope_alto)
            X_test["product_related_duration"] = X_test["product_related_duration"].clip(upper=tope_alto)

            barra = st.progress(0, text="Entrenando baseline...")
            baseline = Pipeline([("prep", crear_preprocessor()),
                                 ("model", DummyClassifier(strategy="stratified", random_state=42))])
            baseline.fit(X_train[cols_modelo], y_train)

            barra.progress(40, text="Entrenando XGBoost sin balanceo...")
            xgb_base = Pipeline([("prep", crear_preprocessor()), ("model", crear_xgb())])
            xgb_base.fit(X_train[cols_modelo], y_train)

            modelos = {"Baseline (aleatorio)": baseline, "XGBoost sin balanceo": xgb_base}
            if tecnica != "Sin balanceo":
                barra.progress(70, text=f"Entrenando XGBoost con {tecnica}...")
                if tecnica == "Random Oversampling":
                    balanceo = RandomOverSampler(sampling_strategy=sampling, random_state=42)
                elif tecnica == "SMOTE":
                    balanceo = SMOTE(sampling_strategy=sampling, k_neighbors=vecinos, random_state=42)
                else:
                    balanceo = RandomUnderSampler(sampling_strategy=sampling, random_state=42)
                xgb_bal = ImbPipeline([("prep", crear_preprocessor()), ("balanceo", balanceo),
                                       ("model", crear_xgb())])
                xgb_bal.fit(X_train[cols_modelo], y_train)
                modelos[f"XGBoost con {tecnica}"] = xgb_bal

            probs = {n: m.predict_proba(X_test[cols_modelo])[:, 1] for n, m in modelos.items()}
            barra.progress(100, text="Listo")

            # el modelo de la demo es el último entrenado
            st.session_state.update({
                "modelos": modelos, "probs": probs, "y_test": y_test,
                "X_test": X_test, "modelo_final": list(modelos)[-1],
            })

        if "modelos" in st.session_state:
            y_test = st.session_state["y_test"]
            st.subheader(f"Resultados en test ({len(y_test):,} sesiones)".replace(",", " "))
            tabla = pd.DataFrame([metricas(n, y_test, p) for n, p in st.session_state["probs"].items()])
            st.dataframe(tabla.round(4), hide_index=True)
            st.success(f"✅ Modelo para la demo: **{st.session_state['modelo_final']}**. "
                       "Ahora ve a **Hacer Predicciones**.")

# ---------------- 3. predicciones ----------------
elif seccion == "🔮 Hacer Predicciones":
    st.header("🔮 Predecir si una sesión termina en compra")

    if "modelos" not in st.session_state:
        st.warning("⚠️ Por favor, entrena primero el modelo en la sección de Entrenamiento.")
    else:
        nombre = st.session_state["modelo_final"]
        modelo = st.session_state["modelos"][nombre]
        st.markdown(f"Modelo usado: **{nombre}**. `revenue` no se ingresa: es lo que el modelo predice.")

        ejemplos = {
            "Sesión 1: recurrente en noviembre, mucho catálogo": dict(
                administrative=3, administrative_duration=90.0, informational=0, informational_duration=0.0,
                product_related=45, product_related_duration=1800.0, page_values=40.0, special_day=0.0,
                month="Nov", visitor_type="Returning_Visitor", weekend=False),
            "Sesión 2: recurrente en diciembre, navegación media": dict(
                administrative=2, administrative_duration=60.0, informational=1, informational_duration=30.0,
                product_related=25, product_related_duration=900.0, page_values=8.0, special_day=0.0,
                month="Dec", visitor_type="Returning_Visitor", weekend=False),
            "Sesión 3: nueva en mayo, pocas páginas": dict(
                administrative=0, administrative_duration=0.0, informational=0, informational_duration=0.0,
                product_related=6, product_related_duration=150.0, page_values=0.0, special_day=0.0,
                month="May", visitor_type="New_Visitor", weekend=True),
        }

        c1, c2 = st.columns([3, 1])
        eleccion = c1.selectbox("Cargar un ejemplo", list(ejemplos.keys()))
        if c2.button("🎲 Sesión al azar de test"):
            fila = st.session_state["X_test"].sample(1).iloc[0]
            st.session_state["aleatoria"] = {c: fila[c] for c in cols_modelo}
            st.session_state["real"] = int(st.session_state["y_test"].loc[fila.name])
        if st.session_state.get("aleatoria") is not None and c2.button("Volver a los ejemplos"):
            st.session_state["aleatoria"] = None

        base = st.session_state.get("aleatoria") or ejemplos[eleccion]
        meses_opc = orden_meses if base["month"] in orden_meses else orden_meses + [base["month"]]
        tipos = ["Returning_Visitor", "New_Visitor", "Other"]

        with st.form("sesion"):
            f1, f2, f3 = st.columns(3)
            with f1:
                month = st.selectbox("Mes", meses_opc, index=meses_opc.index(base["month"]))
                visitor_type = st.selectbox("Tipo de visitante", tipos, index=tipos.index(base["visitor_type"]))
                weekend = st.checkbox("Visita en fin de semana", value=bool(base["weekend"]))
                special_day = st.slider("Cercanía a fecha especial", 0.0, 1.0, float(base["special_day"]), 0.2)
            with f2:
                product_related = st.number_input("Páginas de producto", 0, 800, int(base["product_related"]))
                product_related_duration = st.number_input("Tiempo en páginas de producto (s)", 0.0, 70000.0,
                                                           float(base["product_related_duration"]))
                page_values = st.number_input("page_values", 0.0, 400.0, float(base["page_values"]))
            with f3:
                administrative = st.number_input("Páginas administrativas", 0, 30, int(base["administrative"]))
                administrative_duration = st.number_input("Tiempo en administrativas (s)", 0.0, 4000.0,
                                                          float(base["administrative_duration"]))
                informational = st.number_input("Páginas informativas", 0, 30, int(base["informational"]))
                informational_duration = st.number_input("Tiempo en informativas (s)", 0.0, 3000.0,
                                                         float(base["informational_duration"]))
            predecir = st.form_submit_button("🔮 Predecir", type="primary")

        if predecir:
            sesion = pd.DataFrame([{
                "administrative": administrative, "administrative_duration": administrative_duration,
                "informational": informational, "informational_duration": informational_duration,
                "product_related": product_related, "product_related_duration": product_related_duration,
                "page_values": page_values, "special_day": special_day,
                "month": month, "visitor_type": visitor_type, "weekend": weekend
            }])[cols_modelo]
            prob = float(modelo.predict_proba(sesion)[0, 1])

            r1, r2, r3 = st.columns(3)
            r1.metric("Probabilidad de compra (revenue = True)", f"{prob * 100:.1f}%")
            r2.metric("Probabilidad de no compra (revenue = False)", f"{(1 - prob) * 100:.1f}%")
            r3.metric("Predicción", "Compra" if prob >= 0.5 else "No compra")
            st.progress(prob)
            if prob >= 0.5:
                st.success("✅ Sin cupón: es probable que compre al precio normal.")
            else:
                st.warning("🎟️ Candidata a cupón: es probable que se vaya sin comprar.")
            if st.session_state.get("aleatoria") is not None:
                real = st.session_state["real"]
                st.info(f"Esta sesión es real (de test). Resultado real: **{'Compra' if real else 'No compra'}**.")

# ---------------- 4. análisis ----------------
elif seccion == "📈 Análisis de Resultados":
    st.header("📈 Análisis Detallado de Resultados")

    if "modelos" not in st.session_state:
        st.warning("⚠️ Por favor, entrena primero el modelo en la sección de Entrenamiento.")
    else:
        y_test = st.session_state["y_test"]
        probs = st.session_state["probs"]

        st.subheader("Matrices de confusión en test")
        cols = st.columns(len(probs))
        for col, (n, p) in zip(cols, probs.items()):
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ConfusionMatrixDisplay(confusion_matrix(y_test, (p >= 0.5).astype(int)),
                                   display_labels=["No compra", "Compra"]).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(n, fontsize=10)
            col.pyplot(fig)
            plt.close(fig)

        st.markdown("- **Falso positivo:** el modelo dice compra, no se da cupón y la sesión se va (venta perdida).\n"
                    "- **Falso negativo:** el modelo dice no compra y se da cupón a quien ya iba a comprar (cupón regalado).")

        st.subheader("Curvas ROC y Precision-Recall")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for n, p in probs.items():
            fpr, tpr, _ = roc_curve(y_test, p)
            axes[0].plot(fpr, tpr, label=f"{n} ({roc_auc_score(y_test, p):.4f})")
            prec, rec, _ = precision_recall_curve(y_test, p)
            axes[1].plot(rec, prec, label=f"{n} ({average_precision_score(y_test, p):.4f})")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[0].set_xlabel("Tasa de falsos positivos")
        axes[0].set_ylabel("Recall")
        axes[0].set_title("Curva ROC")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precisión")
        axes[1].set_title("Curva Precision-Recall")
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Variables más importantes (XGBoost sin balanceo)")
        xgb = st.session_state["modelos"]["XGBoost sin balanceo"]
        imp = pd.Series(xgb.named_steps["model"].feature_importances_,
                        index=xgb.named_steps["prep"].get_feature_names_out()).sort_values(ascending=False).head(10)
        st.bar_chart(imp.sort_values(), horizontal=True)
        st.caption("page_values es la variable que más pesa. En el notebook, sin ella el ROC-AUC bajó de 0.93 a 0.75.")
