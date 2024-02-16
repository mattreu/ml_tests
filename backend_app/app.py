from flask import Flask, request, jsonify
from alt_rbm import RBM

app = Flask(__name__)

model = RBM()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # get data from POST request
    recommendations = model.get_recommendations(data)  # make recommendations
    return jsonify({'recommendations': recommendations})  # return recommendations

if __name__ == '__main__':
    app.run(port=5000, debug=True)
