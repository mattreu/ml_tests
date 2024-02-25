from flask import Flask, request, jsonify
from alt_rbm import RBM
from pathlib import Path
import sys

app = Flask(__name__)

rmb_model = RBM()
filename = "rbm_app.json"
rbm_file_path = Path("rbm_models/"+filename)
if not rbm_file_path.is_file():
    sys.exit("No rbm model exists")
else:
    rmb_model.load_from_file(filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # get data from POST request
    recommendations = rmb_model.get_recommendations(data)  # make recommendations
    return jsonify({'recommendations': recommendations})  # return recommendations

if __name__ == '__main__':
    app.run(port=5000, debug=True)
