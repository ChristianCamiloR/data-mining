#importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar el dataset (asegúrate de ajustar la ruta a tu archivo)
df = pd.read_csv('chicago_crimes.csv')

# Convertir la columna 'Date' al formato datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

# Llenar valores nulos
df['Arrest'].fillna(False, inplace=True)
df['District'].fillna(0, inplace=True)
df['Beat'].fillna(0, inplace=True)
df['Primary Type'].fillna('UNKNOWN', inplace=True)
df['Location Description'].fillna('UNKNOWN', inplace=True)

# Extraer día de la fecha
df['Day Date'] = df['Date'].dt.day

# Seleccionar las 6 columnas relevantes
selected_columns = ['Primary Type', 'Location Description', 'Arrest', 'Day Date', 'District', 'Beat']
df = df[selected_columns + ['Domestic']]

# Inicializar y ajustar LabelEncoders
categorical_columns = ['Primary Type', 'Location Description']
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    df[col] = df[col].astype(str)
    df[col] = label_encoders[col].fit_transform(df[col])

# Guardar los LabelEncoders
for col in categorical_columns:
    joblib.dump(label_encoders[col], f'label_encoder_{col.lower().replace(" ", "_")}.pkl')

# Seleccionar características y objetivo
features = df.drop(columns=['Domestic'])
target = df['Domestic']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Hacer predicciones
y_pred = model.predict(X_test_scaled)

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
print('Confusion Matrix:')
print(conf_matrix)

#Guardar el modelo entrenado
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')