from flask import Flask, request, jsonify, render_template
from alt_rbm import RBM
from pathlib import Path
import sys
import os

app = Flask(__name__)

directory = os.path.dirname(os.path.abspath(__file__))
rmb_model = RBM()
filename = "rbm_app.json"
rbm_file_path = Path(os.path.join(directory, "rbm_models", filename))
if not rbm_file_path.is_file():
    sys.exit("No rbm model exists in " + str(rbm_file_path.absolute()))
else:
    rmb_model.load_from_file(filename)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/initial', methods=['POST'])
def initial():
    recommendations = rmb_model.prepare_initial_recommendactions()
    return recommendations

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # get data from POST request
    recommendations = rmb_model.get_recommendations(data)  # make recommendations
    return jsonify({'recommendations': recommendations})  # return recommendations

if __name__ == '__main__':
    app.run(port=5000, debug=True)
