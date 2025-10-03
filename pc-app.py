import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2


USER = st.secrets["postgres"]["USER"]
PASSWORD = st.secrets["postgres"]["PASSWORD"]
HOST = st.secrets["postgres"]["HOST"]
PORT = st.secrets["postgres"]["PORT"]
DBNAME = st.secrets["postgres"]["DBNAME"]

# Configuración de la página
st.set_page_config(page_title="Predicción de Diabetes", page_icon="💉")

# Conectar a la BD
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    cursor = connection.cursor()
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    st.sidebar.success(f"Conectado a la BD. Hora: {result}")
except Exception as e:
    st.sidebar.error(f"Error de conexión: {e}")

# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load("components/diabetes_model.pkl")
        scaler = joblib.load("components/diabetes_scaler.pkl")
        with open("components/diabetes_model_info.pkl", "rb") as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en 'components/'")
        return None, None, None

# Título
st.title("💉 Predicción de progresión de Diabetes")

# Cargar modelo
model, scaler, model_info = load_models()

if model is not None:
    st.header("Ingresa las características del paciente:")

    # Inputs dinámicos según las features
    inputs = []
    for feature in model_info["feature_names"]:
        val = st.number_input(f"{feature}", value=0.0, step=0.1)
        inputs.append(val)

    if st.button("Predecir"):
        # Preparar datos
        features = np.array([inputs])
        features_scaled = scaler.transform(features)

        # Predicción
        prediction = model.predict(features_scaled)[0]

        st.success(f"Predicción de progresión de la diabetes (medida continua): **{prediction:.2f}**")
