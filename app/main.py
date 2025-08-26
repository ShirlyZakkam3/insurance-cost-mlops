from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder="templates")
CORS(app)

model_path = os.path.join(os.path.dirname(__file__), 'insurance_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('frontend.html')  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        age = int(data['age'])
        sex = data['sex'].lower()          
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = data['smoker'].lower()    
        region = data['region'].lower()    

        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }])

        prediction = model.predict(input_df)[0]

        return jsonify({'predicted_cost': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Medical Insurance Cost Prediction API running at http://0.0.0.0:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)
