from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)
application = app

# Memuat model yang telah disimpan dengan exception handling
model = None
model_loading_error = ""

try:
    with open('bagging_classifier.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    model_loading_error = str(e)
    print(f"Error loading the model: {model_loading_error}")

# Route untuk halaman utama (form input data)
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return f"Model loading failed: {model_loading_error}", 500
    
    # Mendapatkan data dari form
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Melakukan prediksi menggunakan model yang telah dilatih
    predictions = [estimator.predict(final_features)[0] for estimator in model['estimators']]
    
    # Majority vote untuk prediksi akhir
    pred_majority_vote = np.bincount(predictions).argmax()
    
    # Menginterpretasi hasil prediksi
    output = 'Class 1' if pred_majority_vote == 1 else 'Class 2'
    
    # Mengarahkan ke halaman hasil prediksi
    return redirect(url_for('result', prediction_text=output, input_data=",".join(map(str, features))))

# Route untuk menampilkan hasil prediksi
@app.route('/result')
def result():
    prediction_text = request.args.get('prediction_text')
    input_data = request.args.get('input_data').split(',')
    input_data = list(map(int, input_data))
    accuracy_model = 0.97  # Sesuaikan dengan akurasi model Anda
    return render_template('result.html', prediction_text=prediction_text, input_data=input_data, accuracy_model=accuracy_model)

if __name__ == '__main__':
    app.run(debug=True)
