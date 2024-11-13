from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = r'modeltitanic.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form submission
        int_features = [
            int(request.form['pclass']),
            1 if request.form['sex'] == 'male' else 0,  # Encoding for 'sex'
            int(request.form['age']),
            int(request.form['sibsp']),
            int(request.form['parch']),
            float(request.form['fare']),  # Adding 'fare' as a float value
            {'C': 0, 'Q': 1, 'S': 2}[request.form['embarked']]  # Encoding for 'embarked'
        ]
        print("int_features:", int_features)  # Debug: Check extracted features
        final_features = [np.array(int_features)]
        print("final_features:", final_features)  # Debug: Check final input array

        # Make prediction
        prediction = model.predict(final_features)
        print("prediction:", prediction)  # Debug: Check prediction output
        output = 'Survived' if prediction[0] == 1 else 'Not Survived'

        # Pass prediction to HTML template
        return render_template('index.html', prediction_text=output)

    except Exception as e:
        print("Error:", e)  # Debug: Print the error message
        return render_template('index.html', prediction_text='There was an error in the prediction process.')

if __name__ == "__main__":
    app.run(debug=True)