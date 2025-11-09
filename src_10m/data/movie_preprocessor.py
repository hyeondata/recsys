"""
Movie data preprocessor for MovieLens 10M dataset.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class MoviePreprocessor:
    """Preprocessor for movie data (movies.dat)"""

    def __init__(self):
        self.genre_list = []
        self.genre_to_idx = {}
        self.movie_id_map = {}
        self.num_movies = 0
        self.num_genres = 0

    def fit(self, movie_file: str) -> 'MoviePreprocessor':
        """
        Fit the preprocessor on movie data.

        Args:
            movie_file: Path to movies.dat (format: movieId::title::genres)

        Returns:
            self
        """
        # Read movies.dat (format: movieId::title::genres)
        movies = []
        with open(movie_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    movie_id = int(parts[0])
                    title = parts[1]
                    genres = parts[2].split('|')
                    movies.append({
                        'movie_id': movie_id,
                        'title': title,
                        'genres': genres
                    })

        self.movies_df = pd.DataFrame(movies)

        # Build movie ID mapping (원본 ID → 연속된 인덱스)
        unique_movie_ids = sorted(self.movies_df['movie_id'].unique())
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movie_ids)}
        self.num_movies = len(unique_movie_ids)

        # Build genre vocabulary
        all_genres = set()
        for genres in self.movies_df['genres']:
            all_genres.update(genres)

        self.genre_list = sorted(list(all_genres))
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genre_list)}
        self.num_genres = len(self.genre_list)

        print(f"MoviePreprocessor fitted:")
        print(f"  - Number of movies: {self.num_movies}")
        print(f"  - Number of genres: {self.num_genres}")
        print(f"  - Genres: {self.genre_list}")

        return self

    def transform_movie_id(self, movie_id: int) -> int:
        """Transform original movie ID to continuous index."""
        return self.movie_id_map.get(movie_id, 0)

    def get_movie_genres(self, movie_id: int) -> np.ndarray:
        """
        Get genre multi-hot encoding for a movie.

        Args:
            movie_id: Original movie ID

        Returns:
            Multi-hot encoded genres (shape: [num_genres])
        """
        movie_row = self.movies_df[self.movies_df['movie_id'] == movie_id]

        genre_vector = np.zeros(self.num_genres, dtype=np.float32)

        if len(movie_row) > 0:
            genres = movie_row.iloc[0]['genres']
            for genre in genres:
                if genre in self.genre_to_idx:
                    genre_vector[self.genre_to_idx[genre]] = 1.0

        return genre_vector

    def fit_transform(self, movie_file: str) -> pd.DataFrame:
        """Fit and transform movie data."""
        self.fit(movie_file)
        return self.movies_df

    def get_info(self) -> Dict:
        """Get preprocessor info."""
        return {
            'num_movies': self.num_movies,
            'num_genres': self.num_genres,
            'genre_list': self.genre_list
        }
