import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MatrixFactorization():
    
    def __init__(self, ratings: np.ndarray, 
                 latent_factors: int, 
                 learning_rate: float = 0.02, 
                 regularization: float = 0.01, 
                 iterations: int = 100, 
                 random_seed:int = None, 
                 momentum:float = 0.2):
        
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

    def get_parameters(self):
        return {
        'latent_factors': self.latent_factors,
        'learning_rate': self.learning_rate,
        'regularization': self.regularization,
        'train_iterations': self.train_iterations,
        'momentum': self.momentum
        }

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
            error = rating - prediction
            # Update biases with momentum
            user_bias_momentum[user_index] = (
                self.momentum * user_bias_momentum[user_index] +
                self.learning_rate * (error - self.regularization * self.user_bias[user_index])
                )
            self.user_bias[user_index] += user_bias_momentum[user_index]
            item_bias_momentum[item_index] = (
                self.momentum * item_bias_momentum[item_index] +
                self.learning_rate * (error - self.regularization * self.item_bias[item_index])
                )
            self.item_bias[item_index] += item_bias_momentum[item_index]

            # Backup of user latent factors to update item latent factors
            user_latent_factors_backup = self.user_latent_factors[user_index, :]

            # Update user and item latent factors matrices with momentum
            user_latent_factors_momentum[user_index, :] = (
                self.momentum * user_latent_factors_momentum[user_index, :] + 
                self.learning_rate * (error * self.item_latent_factors[item_index, :] - 
                                   self.regularization * self.user_latent_factors[user_index, :])
                )
            self.user_latent_factors[user_index, :] += user_latent_factors_momentum[user_index, :]
            item_latent_factors_momentum[item_index, :] = (
                self.momentum * item_latent_factors_momentum[item_index, :] + 
                self.learning_rate * (error * user_latent_factors_backup - 
                                   self.regularization * self.item_latent_factors[item_index, :])
                )
            self.item_latent_factors[item_index, :] += item_latent_factors_momentum[item_index, :]

    def get_root_mean_squared_error(self):
        mse = self.get_mean_squared_error()
        return np.sqrt(mse)

    def get_mean_squared_error(self):
        """
        Returns total mean squared error
        """
        xs, ys = self.rating_matrix.nonzero()
        predicted = self.get_prediction_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.rating_matrix[x, y] - predicted[x, y], 2)
        mse = error / len(xs)
        return mse
    
    def get_rmse_and_similarity(self, ratings):
        """
        Returns various statistics for validation data recommendations
        """
        predicted_ratings, similarities = self.recommend_for_new_users(ratings, return_similarity=True)
        rmses = []
        
        for i in range(len(ratings)):
            non_zero_indices = np.nonzero(ratings[i])
            predicted_ratings_for_user = predicted_ratings[i][non_zero_indices]
            actual_ratings_for_user = ratings[i][non_zero_indices]
            squared_error = np.power(actual_ratings_for_user - predicted_ratings_for_user, 2)
            mse = np.sum(squared_error) / len(actual_ratings_for_user)
            rmse = np.sqrt(mse)
            rmses.append(rmse)
        # RMSE for all users
        mean_rmse = np.mean(rmses)
        return mean_rmse, rmses, similarities
    
    def get_rmse_and_similarity_pm(self, ratings):
        """
        Returns various statistics for validation data recommendations
        """
        predicted_ratings, similarities = self.recommend_for_new_users_pm(ratings, return_similarity=True)
        rmses = []
        
        for i in range(len(ratings)):
            non_zero_indices = np.nonzero(ratings[i])
            predicted_ratings_for_user = predicted_ratings[i][non_zero_indices]
            actual_ratings_for_user = ratings[i][non_zero_indices]
            squared_error = np.power(actual_ratings_for_user - predicted_ratings_for_user, 2)
            mse = np.sum(squared_error) / len(actual_ratings_for_user)
            rmse = np.sqrt(mse)
            rmses.append(rmse)
        # RMSE for all users
        mean_rmse = np.mean(rmses)
        return mean_rmse, rmses, similarities

    def train(self, debug = False):
        # Create initial user and item latent factors matrices using uniform distribution
        random_num_generator = np.random.default_rng(self.random_seed)
        self.user_latent_factors = random_num_generator.uniform(
            low=-1, high=1, size=(self.num_users, self.latent_factors)
            )
        self.item_latent_factors = random_num_generator.uniform(
            low=-1, high=1, size=(self.num_items, self.latent_factors)
            )

        # Init bias
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
            root_mean_squared_error = self.get_root_mean_squared_error()
            training_process.append(root_mean_squared_error)
            if debug:
                print("Iteration: %d error = %.4f" % (i+1, root_mean_squared_error))
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
        prediction: float
        Predicted rating for user
        """
        
        prediction = self.ratings_mean + self.user_bias[user_index] + self.item_bias[item_index] + self.user_latent_factors[user_index, :].dot(self.item_latent_factors[item_index, :].T)
        return prediction
    
    def get_prediction_matrix(self):
        """
        Returns whole prediction matrix
        """

        prediction_matrix = (self.ratings_mean + 
                             self.user_bias[:,np.newaxis] + 
                             self.item_bias[np.newaxis:,] + 
                             self.user_latent_factors.dot(self.item_latent_factors.T))
        return prediction_matrix

    def recommend_for_new_user(self, new_user_ratings):
        """
        Returns recommendations for user

        Parameters
        ----------
        new_user_ratings: ndarray

        Returns
        -------
        most_similar_user_ratings: ndarray
        """
        # Compute similarity between new user and all existing users
        similarity = cosine_similarity(new_user_ratings, self.rating_matrix)

        # Find the most similar user
        most_similar_user_index = np.argmax(similarity)

        # Get the ratings from the most similar user
        most_similar_user_ratings = self.get_prediction_matrix()[most_similar_user_index, :]
        
        return np.array(most_similar_user_ratings)

    def recommend_for_new_users(self, new_users_ratings, return_similarity=False):
        recommendations = []
        highest_similarities = []
        prediction_matrix = self.get_prediction_matrix()

        for new_user_ratings in new_users_ratings:
            # Compute similarity between new user and all existing users
            similarity = cosine_similarity(new_user_ratings.reshape(1, -1), self.rating_matrix)
            
            # Find the most similar user
            most_similar_user_index = np.argmax(similarity)
            if return_similarity:
                highest_similarity = similarity[0, most_similar_user_index]
                highest_similarities.append(highest_similarity)
            
            most_similar_user_ratings = prediction_matrix[most_similar_user_index, :]
            recommendations.append(most_similar_user_ratings)
        
        if return_similarity:
            return np.array(recommendations), np.array(highest_similarities)
        else:
            return np.array(recommendations)
        
    def recommend_for_new_users_pm(self, new_users_ratings, return_similarity=False):
        recommendations = []
        highest_similarities = []
        prediction_matrix = self.get_prediction_matrix()

        for new_user_ratings in new_users_ratings:
            nonzero_mask = new_user_ratings != 0
            masked_new_user_ratings = new_user_ratings[nonzero_mask].reshape(1, -1)
            masked_prediction_matrix = prediction_matrix[:, nonzero_mask]
            similarity = cosine_similarity(masked_new_user_ratings, masked_prediction_matrix)
            
            most_similar_user_index = np.argmax(similarity)
            if return_similarity:
                highest_similarity = similarity[0, most_similar_user_index]
                highest_similarities.append(highest_similarity)
            
            most_similar_user_ratings = prediction_matrix[most_similar_user_index, :]
            recommendations.append(most_similar_user_ratings)
        
        if return_similarity:
            return np.array(recommendations), np.array(highest_similarities)
        else:
            return np.array(recommendations)
