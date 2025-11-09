"""
Dataset class for MovieLens 10M.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict


class MovieRatingDataset(Dataset):
    """
    Dataset for MovieLens 10M ratings.

    Note: ml-10M100K does not have user demographics (age, gender, occupation).
    Only user_id and movie_id + genres are used.
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movie_preprocessor,
        use_genres: bool = True
    ):
        """
        Args:
            ratings_df: DataFrame with columns [user_id, movie_id, rating, timestamp]
            movie_preprocessor: MoviePreprocessor instance
            use_genres: Whether to include genre features
        """
        self.ratings_df = ratings_df.reset_index(drop=True)
        self.movie_preprocessor = movie_preprocessor
        self.use_genres = use_genres

        # Build user ID mapping
        unique_users = sorted(self.ratings_df['user_id'].unique())
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.num_users = len(unique_users)

        print(f"Dataset initialized:")
        print(f"  - Number of ratings: {len(self.ratings_df)}")
        print(f"  - Number of users: {self.num_users}")
        print(f"  - Number of movies: {self.movie_preprocessor.num_movies}")
        print(f"  - Use genres: {self.use_genres}")

    def __len__(self) -> int:
        return len(self.ratings_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.

        Returns:
            user_id: Continuous user index
            movie_id: Continuous movie index
            rating: Rating value (1-5)
            genres: Multi-hot genre vector (if use_genres=True)
        """
        row = self.ratings_df.iloc[idx]

        # Map IDs to continuous indices
        user_id = self.user_id_map[row['user_id']]
        movie_id = self.movie_preprocessor.transform_movie_id(row['movie_id'])
        rating = float(row['rating'])

        if self.use_genres:
            genres = self.movie_preprocessor.get_movie_genres(row['movie_id'])
            return (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(movie_id, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float32),
                torch.tensor(genres, dtype=torch.float32)
            )
        else:
            return (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(movie_id, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float32)
            )

    def get_info(self) -> Dict:
        """Get dataset info."""
        return {
            'num_users': self.num_users,
            'num_movies': self.movie_preprocessor.num_movies,
            'num_genres': self.movie_preprocessor.num_genres if self.use_genres else 0,
            'num_ratings': len(self.ratings_df),
            'rating_stats': {
                'mean': self.ratings_df['rating'].mean(),
                'std': self.ratings_df['rating'].std(),
                'min': self.ratings_df['rating'].min(),
                'max': self.ratings_df['rating'].max()
            }
        }


def load_ratings(ratings_file: str) -> pd.DataFrame:
    """
    Load ratings from ratings.dat.

    Format: userId::movieId::rating::timestamp

    Returns:
        DataFrame with columns [user_id, movie_id, rating, timestamp]
    """
    ratings = []
    with open(ratings_file, 'r') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 4:
                user_id = int(parts[0])
                movie_id = int(parts[1])
                rating = float(parts[2])
                timestamp = int(parts[3])
                ratings.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp
                })

    df = pd.DataFrame(ratings)
    print(f"Loaded {len(df)} ratings from {ratings_file}")
    return df


def train_test_split_temporal(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train/test by timestamp (temporal split).

    Args:
        ratings_df: DataFrame with ratings
        test_ratio: Ratio of test data

    Returns:
        train_df, test_df
    """
    # Sort by timestamp
    sorted_df = ratings_df.sort_values('timestamp').reset_index(drop=True)

    # Split point
    split_idx = int(len(sorted_df) * (1 - test_ratio))

    train_df = sorted_df.iloc[:split_idx].reset_index(drop=True)
    test_df = sorted_df.iloc[split_idx:].reset_index(drop=True)

    print(f"Temporal split:")
    print(f"  - Train: {len(train_df)} ratings")
    print(f"  - Test: {len(test_df)} ratings")

    return train_df, test_df
