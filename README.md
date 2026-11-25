# Predicción de intención de compra en e-commerce

Trabajo final de Business Predictive Analytics (UPC), Grupo 2, sección 9341.

App de Streamlit en 4 secciones: exploración de datos, entrenamiento (con opción de balanceo), predicción de una sesión simulada (probabilidad de `revenue = True`) y análisis de resultados.

- `app.py`: la app. Entrena el modelo desde la app, con los mismos pasos y parámetros del notebook.
- `online_shoppers_intention.csv`: dataset de UCI (Sakar et al., 2019).
- `requirements.txt`: librerías.

Para correrla en local: `pip install -r requirements.txt` y luego `streamlit run app.py`.
