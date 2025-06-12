import streamlit as st
import requests

# Configuración de página
st.set_page_config(page_title="Predicción de Producto credito", layout="centered")
st.title("📈 Predicción de Aceptación de Producto")
st.markdown("Simula el comportamiento de un cliente ante un nuevo producto digital.")

# Inputs del cliente
age = st.slider("🎂 Edad", 25, 70, 40)
monthly = st.number_input("💵 Monto mensual estimado(USD)", min_value=0.0, step=1000.0, value=15000.0)
app_usage = st.slider("📄 Score interno del uso del app", 0, 10, 1)
digital_profile = st.slider("📄 Score de perfil digital", 0, 100, 1)
num_contacts = st.number_input("Numero sincronizados desde el movil", min_value=0.0, step=10.0, value=100.0)
residence_risk = st.selectbox("💼 Zona de residencia", ["Bajo", "Medio", "Alto"])
political_event = st.radio("📬 Si hubo disturbios", ["No", "Sí"])

# Threshold slider
threshold = st.slider("🎚 Umbral de aceptación (threshold)", 0.0, 1.0, 0.5, step=0.01)

# Botón de predicción
if st.button("🔍 Evaluar Probabilidad"):
    with st.spinner("Consultando modelo..."):
        try:
            payload = {
                "age": age,
                "monthly_income_usd": monthly,
                "app_usage_score": app_usage,
                "digital_profile_strength": digital_profile,
                "num_contacts_uploaded": num_contacts,
                "residence_risk_zone": residence_risk,
                "political_event_last_month": 1 if political_event == "Sí" else 0,
                "threshold": threshold
            }

            r = requests.post("http://localhost:8000/predict_credit", json=payload)
            if r.status_code == 200:
                resultado = r.json()
                score = resultado["score_probabilidad"]
                aceptara = resultado["aceptará"]

                st.markdown(f"### 🔢 Score de aceptación: **{score:.3f}**")
                st.markdown(f"### 🎯 Umbral usado: **{threshold:.2f}**")

                if aceptara:
                    st.success("✅ El cliente probablemente **aceptará** el upselling.")
                else:
                    st.warning("⚠️ El cliente probablemente **rechazará** la oferta.")
            else:
                st.error("❌ Error en la respuesta del modelo.")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")


#uvicorn api:app --reload --port 8000
#streamlit run app.py
