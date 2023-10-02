import numpy as np

class RBM:
  
  def __init__(self, ratings: np.ndarray, hidden_nodes_num: int, learning_rate: float = 0.1, iterations: int = 1000):
    """
    Restricted Boltzmann Machine class used for recommendations generation.

    Parameters
    ----------
    ratings : ndarray
        user-item rating matrix used in training
    hidden_nodes_num: int
        number of latent factors
    learning_rate: float
    iterations: int
        number of model training iterations
    """

    self.hidden_nodes_num = hidden_nodes_num
    self.training_samples, self.visible_nodes_num = ratings.shape
    self.debug_print = True
    self.ratings = ratings
    self.learning_rate = learning_rate
    self.iterations = iterations

    random_generator = np.random.RandomState(1234)

    # Create weight matrix (visible_nodes_num x hidden_nodes_num)
    # Uniform distribution where all states are equally likely to appear
    self.weights = np.asarray(random_generator.uniform(
			low=-0.1 * np.sqrt(6. / (hidden_nodes_num + self.visible_nodes_num)),
      high=0.1 * np.sqrt(6. / (hidden_nodes_num + self.visible_nodes_num)),
      size=(self.visible_nodes_num, self.hidden_nodes_num)))

    # Add bias into the first row and first column
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self):
    """
    Restricted Boltzmann Machine class used for recommendations generation.
    """
    # Insert bias 1 into the first column of training data
    data = np.insert(self.ratings, 0, 1, axis = 1)

    for iteration in range(self.iterations):
      # Activate hidden nodes then clamp probabilities using logistic function
      pos_hidden_activations = np.dot(data, self.weights)
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fix bias
      # Activated hidden nodes
      pos_hidden_states = pos_hidden_probs > np.random.rand(self.training_samples, self.hidden_nodes_num + 1)
      # Measure whether both nodes are active (a measure of how much the input and hidden layer agree)
      pos_associations = np.dot(data.T, pos_hidden_probs)

      # Reconstruct visible nodes and sample again from the hidden nodes
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights
      self.weights += self.learning_rate * ((pos_associations - neg_associations) / self.training_samples)

      error = np.sum((data - neg_visible_probs) ** 2)
      if self.debug_print:
        print("Iteration %s: error is %s" % (iteration, error))

  def run_visible(self, data):
    """
    After RBM has been trained use a set of visible nodes, to generate a sample of hidden nodes.
    
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
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.hidden_nodes_num + 1)
  
    # Take states without bias
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  def run_hidden(self, data):
    """
    After RBM has been trained use a set of hidden nodes, to generate a sample of visible nodes.

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
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.visible_nodes_num + 1)

    # Take states without bias
    visible_states = visible_states[:,1:]
    return visible_states
  
  def get_recommendations(self, data):
    recommendations = self.run_hidden(self.run_visible(data))
    return recommendations

  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
  training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]]) #todo load data from file
  model = RBM(training_data, 5)
  model.train()
  print(model.weights)
  user = np.array([[0,0,0,1,1,0]])
  print(model.get_recommendations(user))
