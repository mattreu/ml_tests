import numpy as np

class Matrix_Factorization():
    
    def __init__(self, ratings: np.ndarray, latent_factors: int, learning_rate: float = 0.1, regularization: float = 0.01, iterations: int = 100, random_seed:int = None, momentum:float = 0.2) -> None:
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Parameters
        ----------
        ratings : ndarray
            user-item rating matrix.
        latent_factors: int
            number of latent factors
        learning_rate: float
        regularization: float
            regularization parameter to avoid overfitting
        iterations: int
            number of model training iterations
        momentum: float
            (optimization) momentum value adds portion of previous movement to current movement so learning is faster
        """
        
        self.rating_matrix = ratings
        self.num_users, self.num_items = ratings.shape
        self.latent_factors = latent_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.train_iterations = iterations
        self.random_seed = random_seed
        self.momentum = momentum
    
    def run_sgd(self):
        """
        Execute stochastic gradient descent algorithm to minimize prediction error
        """
        # Initialize momentum for biases and latent factors
        user_bias_momentum = np.zeros_like(self.user_bias)
        item_bias_momentum = np.zeros_like(self.item_bias)
        user_latent_factors_momentum = np.zeros_like(self.user_latent_factors)
        item_latent_factors_momentum = np.zeros_like(self.item_latent_factors)
    
        for user_index, item_index, rating in self.training_samples:
            # Predict rating then difference between it and actual rating
            prediction = self.get_rating(user_index, item_index)
            error = (rating - prediction)
    
            # Update biases with momentum
            user_bias_momentum[user_index] = self.momentum * user_bias_momentum[user_index] + self.learning_rate * (error - self.regularization * self.user_bias[user_index])
            self.user_bias[user_index] += user_bias_momentum[user_index]
            item_bias_momentum[item_index] = self.momentum * item_bias_momentum[item_index] + self.learning_rate * (error - self.regularization * self.item_bias[item_index])
            self.item_bias[item_index] += item_bias_momentum[item_index]
    
            # Backup of user latent factors for it needs update but these values also update item latent factors
            user_latent_factors_backup = self.user_latent_factors[user_index, :][:]
    
            # Update user and item latent factors matrices with momentum
            user_latent_factors_momentum[user_index, :] = self.momentum * user_latent_factors_momentum[user_index, :] + self.learning_rate * (error * self.item_latent_factors[item_index, :] - self.regularization * self.user_latent_factors[user_index,:])
            self.user_latent_factors[user_index, :] += user_latent_factors_momentum[user_index, :]
            item_latent_factors_momentum[item_index, :] = self.momentum * item_latent_factors_momentum[item_index, :] + self.learning_rate * (error * user_latent_factors_backup - self.regularization * self.item_latent_factors[item_index,:])
            self.item_latent_factors[item_index, :] += item_latent_factors_momentum[item_index, :]

    def get_mean_squared_error(self):
        """
        Returns total mean squared error
        """
        xs, ys = self.rating_matrix.nonzero()
        predicted = self.get_prediction_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.rating_matrix[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def train(self):
        # Create initial user and item latent factors matrices using normal distribution
        random_num_generator = np.random.default_rng(self.random_seed)
        self.user_latent_factors = random_num_generator.normal(scale=1./self.latent_factors, size=(self.num_users, self.latent_factors))
        self.item_latent_factors = random_num_generator.normal(scale=1./self.latent_factors, size=(self.num_items, self.latent_factors))
        
        # Initialize biases
        self.user_bias = np.zeros(self.num_users)
        self.item_bias = np.zeros(self.num_items)
        self.ratings_mean = np.mean(self.rating_matrix[np.where(self.rating_matrix != 0)])
        
        # Create a list of training samples
        self.training_samples = [
            (user, item, self.rating_matrix[user, item])
            for user in range(self.num_users)
            for item in range(self.num_items)
            if self.rating_matrix[user, item] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.train_iterations):
            random_num_generator.shuffle(self.training_samples)
            self.run_sgd()
            mean_squared_error = self.get_mean_squared_error()
            training_process.append((i, mean_squared_error))
            # (optional) Print training progress every 10th iteration
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mean_squared_error))
        
        return training_process

    def get_rating(self, user_index: int, item_index: int):
        """
        Returns predicted rating of item by user

        Parameters
        ----------
        user_index : int
        item_index: int

        Returns
        -------
        prediction: list
        List with predicted values
        """
        prediction = self.ratings_mean + self.user_bias[user_index] + self.item_bias[item_index] + self.user_latent_factors[user_index, :].dot(self.item_latent_factors[item_index, :].T)
        return prediction
    
    def get_prediction_matrix(self):
        """
        Returns whole prediction matrix
        """
        prediction_matrix = self.ratings_mean + self.user_bias[:,np.newaxis] + self.item_bias[np.newaxis:,] + self.user_latent_factors.dot(self.item_latent_factors.T)
        return prediction_matrix
