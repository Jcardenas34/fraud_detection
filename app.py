from flask import Flask, jsonify, request
from flask import render_template
import numpy as np
from fraud_detection.core_detection import detect_fraud
from fraud_detection.helpers import load_data_subset


app = Flask(__name__)

# @app.route('/')
# def inference():

#     return "<p>Welcome to the Model Inference API!</p>"
    # data = request.get_json()
    # if not data or 'input' not in data:
    #     return jsonify({'error': 'Invalid input'}), 400
    
    # input_data = data['input']
    
    # # Here you would typically call your model's inference function
    # # For demonstration, we'll just echo the input back
    # output_data = {'output': f'Processed {input_data}'}
    
    # return jsonify(output_data)

@app.route('/page/<int:page_id>')
def launch_page(page_id):
    return f"<p>Welcome to launch page {page_id}!!</p>"

@app.route('/')
def webpage():
    return render_template('index.html')    


@app.route('/input/')
def input_data():
    return render_template('input.html')

@app.route('/submit/', methods=['POST'])
def submit():
    name = request.form['name']
    age = request.form['age']
    return render_template('result.html', name=name, age=age)

@app.route('/data/')
def data_table():
    # Send dataset to HTML
    return render_template('data_selection.html', data=df.to_dict(orient='records'))


@app.route('/predict/', methods=['POST'])
def predict():

    data_path = "./dataset/synthetic_dataset.h5"
    data_points = load_data_subset(data_path=data_path, n_samples=-1)


    model_path = './models/fraud_autoencoder_model.h5'
    data = request.form['sample_input']
    # Assuming data is a string representation of the input data
    # Convert the input data to the expected format (e.g., list or array)
    try:
        data = [float(x) for x in data.split(',')]
        data = np.array(data)
    except ValueError as e:
        return render_template('error.html', error=str(e))

    fraud_dict = detect_fraud(model_path, data,  threshold=2.0)
    print(fraud_dict["fraud"], fraud_dict["zscore"])
    return render_template('result.html', 
                           fraud=fraud_dict["fraud"], 
                           zscore=fraud_dict["zscore"])

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)