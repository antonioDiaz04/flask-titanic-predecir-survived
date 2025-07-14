from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelos
try:
    random_forest_model = joblib.load('random_forest_model.pkl')
    encoder = joblib.load('ordinal_encoder.pkl')
    scaler = joblib.load('robust_scaler.pkl')
    pca = joblib.load('pca.pkl')
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Title', 'FamilySize']
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
        title = data['name'].split(',')[1].split('.')[0].strip()
        if title in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev','Sir','Jonkheer','Dona']:
            title = 'Rare'
        elif title in ['Mlle', 'Ms']:
            title = 'Miss'
        elif title == 'Mme':
            title = 'Mrs'

        family_size = data['sibsp'] + data['parch'] + 1

        df = pd.DataFrame([{
            'Pclass': data['pclass'],
            'Name': data['name'],
            'Sex': data['sex'],
            'Age': data['age'],
            'SibSp': data['sibsp'],
            'Parch': data['parch'],
            'Ticket': data['ticket'],
            'Fare': data['fare'],
            'Title': title,
            'FamilySize': family_size
        }])

        df[['Sex', 'Title', 'Ticket']] = encoder.transform(df[['Sex', 'Title', 'Ticket']])
        X_input = df[features].copy()
        X_scaled = scaler.transform(X_input)
        X_pca = pca.transform(X_scaled)

        prediction = random_forest_model.predict(X_pca)
        survived = int(prediction[0])

        return jsonify({'survived': survived})

    except Exception as e:
        app.logger.error(f'Error en predicción: {e}')
        return jsonify({'error': 'Error al procesar predicción'}), 500

if __name__ == '__main__':
    app.run(debug=True)
