import zipfile
import pandas as pd

class data_provider:
    def __init__(self) -> None:
        zip_ref = zipfile.ZipFile("ml-100k.zip", "r")
        zip_ref.extractall()
        print(zip_ref.read('ml-100k/u.info'))

    # Utility to split the data into training and test sets.
    def split_dataframe(self, df: pd.DataFrame, holdout_fraction=0.1):
        """
        Splits a DataFrame into training and test sets.

        Parameters
        ----------
        df: DataFrame
            Data to split
        holdout_fraction: float
            Fraction of dataframe rows to use in the test set
        
        Returns
        -------
        train: DataFrame
            Training data
        test: DataFrame
            Testing data
        """
        test = df.sample(frac=holdout_fraction, replace=False)
        train = df[~df.index.isin(test.index)]
        return train, test

    def get_ratings(self):
        ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        ratings = pd.read_csv(
            'ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')
        
        # Shift id to start at 0
        ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x-1))
        ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x-1))
        ratings["rating"] = ratings["rating"].apply(lambda x: float(x))
        return ratings
    
    def get_movies(self):
        # The movies file contains a binary feature for each genre.
        genre_cols = [
            "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        movies_cols = [
            'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
        ] + genre_cols
        movies = pd.read_csv(
            'ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')
        
        # Shift id to start at 0
        movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x-1))

        # Get only year
        movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
        
        return movies
    
    def get_users(self):
        users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        users = pd.read_csv(
            'ml-100k/u.user', sep='|', names=users_cols, encoding='latin-1')
        
        # Shift id to start at 0
        users["user_id"] = users["user_id"].apply(lambda x: str(x-1))

        return users