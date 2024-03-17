import numpy as np
from pathlib import Path
import json
import os

class RBM:
  
  def __init__(self, ratings: np.ndarray = np.asarray([]), hidden_nodes_num: int = 10, learning_rate: float = 0.1, iterations: int = 1000, random_seed: int = None):
    """
    Restricted Boltzmann Machine class used for recommendations generation

    Parameters
    ----------
    ratings : ndarray
        user-item rating matrix used in training
    hidden_nodes_num: int
        number of latent factors
    learning_rate: float
    iterations: int
        number of model training iterations
    random_seed: int
        seed for random numbers generation used for testing purposes
    """

    self.hidden_nodes_num = hidden_nodes_num
    self.debug_print = False
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.generator = np.random.default_rng(random_seed)
    self.ratings = self.visible_nodes_num = self.training_samples = None
    self.load_ratings(ratings)

  def set_debug(self, debug_print):
    """
    Set outputting debug messages about model's work

    Parameters
    ----------
    debug_print : bool
      If True model will output additional info when working
    """
    self.debug_print = debug_print

  def get_learning_rate(self):
    return self.learning_rate

  def load_ratings(self, ratings:np.ndarray):
    """
    Load ratings matrix used to train model

    Parameters
    ----------
    ratings : ndarray

    Returns
    -------
    Returns 0 if provided ratings are not compatible with actual model weights (if model was trained before)
    """
    if ratings.shape[0]!=0:
      if ((self.visible_nodes_num!=None) & (ratings.shape[1]!=self.visible_nodes_num)):
        return 0
      self.ratings = ratings
      self.training_samples, self.visible_nodes_num = ratings.shape
      # Create weight matrix (visible_nodes_num x hidden_nodes_num)
      # Uniform dist -> all states are equally likely to appear
      self.weights = self.generator.uniform(
        low=-0.1 * np.sqrt(6. / (self.hidden_nodes_num + self.visible_nodes_num)),
        high=0.1 * np.sqrt(6. / (self.hidden_nodes_num + self.visible_nodes_num)),
        size=(self.visible_nodes_num, self.hidden_nodes_num))

      # Add bias into first row and first column
      self.weights = np.insert(self.weights, 0, 0, axis = 0)
      self.weights = np.insert(self.weights, 0, 0, axis = 1)
    else:
      self.training_samples = 0
  
  def load_from_file(self, filename:str):
    directory = os.path.dirname(os.path.abspath(__file__))
    path = Path(os.path.join(directory, "rbm_models", filename))
    if path.is_file():
        with path.open('r') as file:
          data = json.load(file)
          self.weights = np.array(data['weights'])
          self.visible_nodes_num = data['visible_nodes_num']
          self.hidden_nodes_num = data['hidden_nodes_num']
          self.load_ratings(np.asarray([])) # load empty ratings to avoid errors with number of visible nodes
    else:
        print(f"No such file: '{path}'")
  
  def save_to_file(self, filename:str):
    directory = os.path.dirname(os.path.abspath(__file__))
    path = Path(os.path.join(directory, "rbm_models", filename))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'weights': self.weights.tolist(),
        'visible_nodes_num': self.visible_nodes_num,
        'hidden_nodes_num': self.hidden_nodes_num
    }
    with path.open('w') as file:
        json.dump(data, file)

  def train(self, return_error: bool = False):
    """
    Run model training

    Parameters
    ----------
    return_error : bool
      If True function will return list with rmse value for each iteration

    Returns
    -------
    Returns 0 if no ratings for training were provided
    iter_error: list
      Error value for each iteration list
    """

    if self.training_samples==0:
      return 0
    
    # Insert bias 1 into first column of training data
    data = np.insert(self.ratings, 0, 1, axis = 1)
    iter_error = []

    for iteration in range(self.iterations):
      # Activate hidden nodes then clamp probabilities using logistic function
      pos_hidden_activations = np.dot(data, self.weights)
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fix bias
      # Activated hidden nodes
      pos_hidden_states = pos_hidden_probs > self.generator.random(size=(self.training_samples, self.hidden_nodes_num + 1))
      # Measure whether both nodes are active (a measure of how much the input and hidden layer agree)
      pos_associations = np.dot(data.T, pos_hidden_probs)

      # Reconstruct visible nodes and sample again from hidden nodes
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights
      self.weights += self.learning_rate * ((pos_associations - neg_associations) / self.training_samples)

      error = np.sum((data - neg_visible_probs) ** 2)
      if return_error:
        iter_error.append(error)
      if self.debug_print:
        print("Iteration %s: error is %s" % (iteration, error))
    if return_error:
      return iter_error

  def run_visible(self, data):
    """
    After RBM has been trained use a set of visible nodes, to generate a sample of hidden nodes
    
    Parameters
    ----------
    data: ndarray
      Visible nodes states matrix
    
    Returns
    -------
    hidden_states: ndarray
      Activated hidden nodes matrix
    """
    
    num_examples = data.shape[0]
    
    # Hidden nodes matrix (with bias)
    hidden_states = np.ones((num_examples, self.hidden_nodes_num + 1))
    
    # Insert bias 1 into data
    data = np.insert(data, 0, 1, axis = 1)

    # Clamped hidden nodes activation probabilities
    hidden_activations = np.dot(data, self.weights)
    hidden_probs = self._logistic(hidden_activations)
    # Activate hidden nodes
    hidden_states[:,:] = hidden_probs > self.generator.random(size=(num_examples, self.hidden_nodes_num + 1))
  
    # Take states without bias
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  def run_hidden(self, data):
    """
    After RBM has been trained use a set of hidden nodes, to generate a sample of visible nodes

    Parameters
    ----------
    data: ndarray
      Hidden nodes states matrix

    Returns
    -------
    visible_states: ndarray
      Activated visible nodes matrix
    """

    num_examples = data.shape[0]

    # Visible nodes matrix (with bias)
    visible_states = np.ones((num_examples, self.visible_nodes_num + 1))

    # Insert bias 1 into data
    data = np.insert(data, 0, 1, axis = 1)

    # Clamped visible nodes activation probabilities
    visible_activations = np.dot(data, self.weights.T)
    visible_probs = self._logistic(visible_activations)
    # Activate hidden nodes
    visible_states[:,:] = visible_probs > self.generator.random(size=(num_examples, self.visible_nodes_num + 1))

    # Take states without bias
    visible_states = visible_states[:,1:]
    return visible_states

  def prepare_initial_recommendations(self):
    recommendations_raw = []
    
    # Get some recommendations from each hidden node
    for hidden_node in range(self.hidden_nodes_num):
      nodes = np.zeros((1,self.hidden_nodes_num))
      nodes[0,hidden_node] = 1
      recommendations_raw.append(np.nonzero(self.run_hidden(nodes)[0])[0])
    
    # Choose part of recommendations to display
    movies_per_node = 10
    recommendations = []
    for index in range(self.hidden_nodes_num):
      if len(recommendations_raw[index]) > 10:
        recommendations.append(np.random.choice(recommendations_raw[index], movies_per_node, replace=False))
      else:
        recommendations.append(recommendations_raw[index])
      # Delete movies that appear in previous sets of recommendations
      if index != self.hidden_nodes_num-1:
        for raw_index in range(index + 1, self.hidden_nodes_num):
          recommendations_raw[raw_index] = np.setdiff1d(recommendations_raw[raw_index], recommendations[index], True)

    return recommendations
  
  def get_initial_recommendations(self, data):
    user_choices = np.zeros((1,self.visible_nodes_num))
    for movie_id in data:
      user_choices[0, int(movie_id)] = 1
    return np.nonzero(self.get_recommendations(np.array(user_choices))[0])[0]
  
  def get_recommendations(self, data):
    """
    Run visible and hidden nodes to generate recommendations

    Parameters
    ----------
    data: ndarray
      User's current watched/liked movies (visible states matrix)

    Returns
    -------
    recommendations: ndarray
      Recommendations matrix
    """
    recommendations = self.run_hidden(self.run_visible(data))
    return recommendations

  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

# sample usage
# if __name__ == '__main__':
#   training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
#   model = RBM(training_data, 5)
#   model.train()
#   print(model.weights)
#   user = np.array([[0,0,0,1,1,0]])
#   test = [1,2,3]
#   print(model.get_initial_recommendations(test))
#   print(model.get_recommendations(user))
#   print(model.prepare_initial_recommendations())
