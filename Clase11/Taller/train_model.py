import pandas as pd
from pycaret.classification import *
import mlflow
#import mlflow.pycaret

# Cargar dataset
df = pd.read_csv("fintech_credit_approval.csv")

# Preparar datos
df_model = df.drop(columns=["user_id"])

# Configurar MLflow local
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("credit_approval")

# Setup de PyCaret
s = setup(
    data=df_model,
    target="approved",
    session_id=606,
    log_experiment=True,
    experiment_name="credit_approval",
    verbose=True,
    profile=False,
    use_gpu=False
)

# Entrenar modelo y seleccionar el mejor
best_model = compare_models()

# Evaluar con visualizaciones: ROC, PR, matriz confusión, SHAP
evaluate_model(best_model)

# Registrar modelo en MLflow
#mlflow.pycaret.log_model(best_model, "upsell_model")

# Guardar localmente
save_model(best_model, "credit_model")

print("✅ Modelo de upsell entrenado, evaluado y registrado.")

#mlflow ui --backend-store-uri file:./mlruns --port 5000
#python train_model.py
