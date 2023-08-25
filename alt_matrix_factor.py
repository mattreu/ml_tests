import numpy as np

class Matrix_Factorization():
    
    def __init__(self, ratings, latent_factors, learning_rate, regularization, iterations) -> None:
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
        """
        
        self.rating_matrix = ratings
        self.num_users, self.num_items = ratings.shape
        self.latent_factors = latent_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.train_iterations = iterations

    def get_rating(self, user_index, item_index):
        """
        Returns predicted rating of item by user
        """
        prediction = self.ratings_mean + self.user_bias[user_index] + self.item_bias[item_index] + self.user_latent_factors[user_index, :].dot(self.item_latent_factors[item_index, :].T)
        return prediction
    
    def get_prediction_matrix(self):
        """
        Returns whole prediction matrix
        """
        prediction_matrix = self.ratings_mean + self.user_bias[:,np.newaxis] + self.item_bias[np.newaxis:,] + self.user_latent_factors.dot(self.item_latent_factors.T)
        return prediction_matrix
    
    def run_sgd(self):
        """
        Execute stochastic gradient descent algorithm to minimize prediction error
        """
        for user_index, item_index, rating in self.training_samples:
            # Predict rating then difference between it and actual rating
            prediction = self.get_rating(user_index, item_index)
            error = (rating - prediction)
            
            # Update biases
            self.user_bias[user_index] += self.learning_rate * (error - self.regularization * self.user_bias[user_index])
            self.item_bias[item_index] += self.learning_rate * (error - self.regularization * self.item_bias[item_index])
            
            # Backup of user latent factors for it needs update but these values also update item latent factors
            user_latent_factors_backup = self.user_latent_factors[user_index, :][:]
            
            # Update user and item latent factors matrices
            self.user_latent_factors[user_index, :] += self.learning_rate * (error * self.item_latent_factors[item_index, :] - self.regularization * self.user_latent_factors[user_index,:])
            self.item_latent_factors[item_index, :] += self.learning_rate * (error * user_latent_factors_backup - self.regularization * self.item_latent_factors[item_index,:])

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
        random_num_generator = np.random.default_rng()
        self.user_latent_factors = random_num_generator.normal(scale=1./self.latent_factors, size=(self.num_users, self.latent_factors))
        self.item_latent_factors = np.random.normal(scale=1./self.latent_factors, size=(self.num_items, self.latent_factors))
        
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
            np.random.shuffle(self.training_samples)
            self.run_sgd()
            mean_squared_error = self.get_mean_squared_error()
            training_process.append((i, mean_squared_error))
            # (optional) Print training progress every 10th iteration
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mean_squared_error))
        
        return training_process
    