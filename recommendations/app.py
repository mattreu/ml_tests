from flask import Flask, request, jsonify, render_template, current_app
from alt_rbm import RBM
from data_provider import data_provider
from pathlib import Path
import sys
import os
import glob

app = Flask(__name__)
with app.app_context():
    current_app.rbm_model = RBM()

provider = data_provider()
movies = provider.get_movies()

def get_movies_data(movie_ids):
    data = movies.iloc[movie_ids]
    return data.to_dict('records')

def get_models():
    directory = os.path.dirname(os.path.abspath(__file__))
    rbm_models_directory = os.path.join(directory, "rbm_models")
    return [os.path.splitext(os.path.basename(filepath))[0] for filepath in glob.glob(rbm_models_directory + "/*")]

@app.route('/')
def home():
    models = get_models()
    context = {'models': models}
    return render_template('home.html', context=context)

@app.route('/load_rbm_model', methods=['POST'])
def load_rbm_model():
    data = request.get_json()
    directory = os.path.dirname(os.path.abspath(__file__))
    current_app.rbm_model = RBM()
    filename = data['model'] + '.json'
    rbm_file_path = Path(os.path.join(directory, "rbm_models", filename))
    if not rbm_file_path.is_file():
        return jsonify({'success': False, 'message': "Brak modelu w: " + str(rbm_file_path.absolute())})
    else:
        current_app.rbm_model.load_from_file(filename)
    return jsonify({'success': True, 'message': "Model poprawnie wczytany"})

# actions for chosen model(s)
@app.route('/actions')
def actions():
    return render_template('actions.html')

# initial data gathering to recommend to new user
@app.route('/initial', methods=['POST'])
def initial():
    recommendations = current_app.rbm_model.prepare_initial_recommendations()
    recommendations = [get_movies_data(movie_ids) for movie_ids in recommendations if movie_ids.size > 0]
    context = {'recommendations': recommendations}
    return render_template('initial.html', context=context)

# test users to choose from
@app.route('/prepared', methods=['POST'])
def prepared():
    return render_template('prepared.html')

# get recommendations
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    recommendations = current_app.rbm_model.get_recommendations(data)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
