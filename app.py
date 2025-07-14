from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelos y transformadores
try:
    random_forest_model = joblib.load('random_forest_model.pkl')
    encoder = joblib.load('ordinal_encoder.pkl')
    scaler = joblib.load('robust_scaler.pkl')
    pca = joblib.load('pca_model.pkl')

    # Estas deben coincidir exactamente con las columnas usadas en entrenamiento
    # Se ha corregido la lista 'features' para que coincida con la del notebook.
    features = ['Sex', 'Ticket', 'Age', 'Fare', 'Pclass', 'SibSp', 'Title', 'FamilySize']

    app.logger.info("Modelos cargados correctamente.")
except Exception as e:
    random_forest_model = None
    app.logger.error(f"Error cargando modelos: {e}")

@app.route('/')
def titanic_page():
    return render_template('index.html')

@app.route('/api/predict-titanic-survival', methods=['POST'])
def predict_titanic_survival():
    if not random_forest_model:
        return jsonify({'error': 'Modelo no disponible'}), 500

    try:
        data = request.json

        # Validar que 'name' tenga formato correcto
        name_parts = data['name'].split(',')
        if len(name_parts) < 2 or '.' not in name_parts[1]:
            raise ValueError("Formato de nombre incorrecto. Usa: Apellido, Título. Nombre")

        title = name_parts[1].split('.')[0].strip()

        # Reemplazo de títulos raros
        if title in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev','Sir','Jonkheer','Dona']:
            title = 'Rare'
        elif title in ['Mlle', 'Ms']:
            title = 'Miss'
        elif title == 'Mme':
            title = 'Mrs'

        family_size = data['sibsp'] + data['parch'] + 1

        df = pd.DataFrame([{
            'Pclass': data['pclass'],
            'Name': data['name'],  # No usada en features pero útil para depurar
            'Sex': data['sex'],
            'Age': data['age'],
            'SibSp': data['sibsp'],
            'Parch': data['parch'],
            'Ticket': data['ticket'],
            'Fare': data['fare'],
            'Title': title,
            'FamilySize': family_size
        }])

        # Codificar variables categóricas
        # Asegúrate que 'Ticket' sea una columna reconocida por el encoder si no lo era antes
        # Si tu encoder fue entrenado sin 'Ticket' para codificación ordinal, esto podría dar un error.
        # Basado en el notebook, 'Ticket' sí se codificaba.
        df[['Sex', 'Title', 'Ticket']] = encoder.transform(df[['Sex', 'Title', 'Ticket']])


        # Extraer solo las columnas de features en el orden correcto
        X_input = df[features].copy()
        X_scaled = pd.DataFrame(scaler.transform(X_input), columns=features)
        X_pca = pca.transform(X_scaled)

        # Predicción
        prediction = random_forest_model.predict(X_pca)
        survived = int(prediction[0])

        return jsonify({'survived': survived})

    except Exception as e:
        app.logger.error(f'Error en predicción: {e}')
        return jsonify({'error': 'Error al procesar predicción'}), 500

if __name__ == '__main__':
    app.run(debug=True)