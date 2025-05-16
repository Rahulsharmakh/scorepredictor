import os
import joblib
import pandas as pd
from django.shortcuts import render

# Set base directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'predictor', 'model.pkl')
TRANSFORMER_PATH = os.path.join(BASE_DIR, 'predictor', 'column_transformer.pkl')

# Load model and transformer
model = joblib.load(MODEL_PATH)
ct = joblib.load(TRANSFORMER_PATH)

# Label encodings used during training
label_encodings = {
    'gender': {'female': 0, 'male': 1},
    'lunch': {'standard': 1, 'free/reduced': 0},
    'test preparation course': {'none': 0, 'completed': 1}
}

def predict_score(request):
    if request.method == 'POST':
        try:
            gender = request.POST['gender']
            race = request.POST['race']
            education = request.POST['education']
            lunch = request.POST['lunch']
            test_prep = request.POST['test preparation course']
            math_score = int(request.POST['math'])
            reading_score = int(request.POST['reading'])

            # Create DataFrame for prediction
            input_data = pd.DataFrame([[  
                label_encodings['gender'][gender],
                race,
                education,
                label_encodings['lunch'][lunch],
                label_encodings['test preparation course'][test_prep],
                math_score,
                reading_score
            ]], columns=[
                'gender', 'race/ethnicity', 'parental level of education', 
                'lunch', 'test preparation course', 'math score', 'reading score'
            ])

            # Transform input and predict
            input_transformed = ct.transform(input_data)
            prediction = model.predict(input_transformed)[0]
            prediction = float(f"{prediction:.8f}")  

            return render(request, 'form.html', {'result': prediction})
        
        except Exception as e:
            return render(request, 'form.html', {'result': f'Error: {str(e)}'})

    return render(request, 'form.html')
