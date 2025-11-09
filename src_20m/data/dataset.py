"""
ml-20m 영화 추천을 위한 데이터셋 클래스
사용자 demographic 정보 없음 (user_id, movie_id, rating만 사용)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
import numpy as np

class MovieRatingDataset(Dataset):
    """ml-20m 영화 제목과 평점 데이터셋 (사용자 특성 없음)"""

    def __init__(
        self,
        movie_data: pd.DataFrame,
        rating_data: pd.DataFrame,
        vocab: Dict[str, int],
        max_length: int = 10
    ):
        """
        Args:
            movie_data: 전처리된 영화 데이터
            rating_data: 평점 데이터 (userId, movieId, rating, timestamp)
            vocab: 단어 사전
            max_length: 제목 최대 길이
        """
        self.movie_data = movie_data
        self.ratings = rating_data
        self.vocab = vocab
        self.max_length = max_length

        # 데이터 준비
        self.prepare_data()

    def prepare_data(self):
        """평점 데이터와 영화 데이터 결합"""
        # 영화 정보와 결합
        self.data = self.ratings.merge(
            self.movie_data[['movieId', 'clean_title', 'encoded_title']],
            on='movieId',
            how='inner'
        )

        print(f"  데이터셋 크기: {len(self.data)}")
        print(f"  사용자 수: {self.data['userId'].nunique()}")
        print(f"  영화 수: {self.data['movieId'].nunique()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        # 인코딩된 제목
        encoded_title = torch.tensor(row['encoded_title'], dtype=torch.long)

        # 평점 (원본 rating 그대로 사용)
        rating = torch.tensor(row['rating'], dtype=torch.float32)

        # 사용자 ID와 영화 ID (1-based indexing 유지, 임베딩에서 처리)
        user_id = torch.tensor(row['userId'], dtype=torch.long)
        movie_id = torch.tensor(row['movieId'], dtype=torch.long)

        return {
            'encoded_title': encoded_title,
            'rating': rating,
            'user_id': user_id,
            'movie_id': movie_id
        }


def load_and_split_ratings(
    data_dir: str,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ml-20m ratings.csv를 train/test로 분할

    Args:
        data_dir: ml-20m 데이터 디렉토리
        train_ratio: 학습 데이터 비율
        random_seed: 랜덤 시드

    Returns:
        train_df, test_df
    """
    import os

    ratings_path = os.path.join(data_dir, 'ratings.csv')
    print(f"평점 데이터 로드 중: {ratings_path}")

    ratings_df = pd.read_csv(ratings_path)
    print(f"  총 평점 수: {len(ratings_df):,}")

    # 랜덤 셔플 후 분할
    ratings_df = ratings_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    split_idx = int(len(ratings_df) * train_ratio)
    train_df = ratings_df[:split_idx]
    test_df = ratings_df[split_idx:]

    print(f"  Train: {len(train_df):,} ({train_ratio*100:.1f}%)")
    print(f"  Test: {len(test_df):,} ({(1-train_ratio)*100:.1f}%)")

    return train_df, test_df


def create_stratified_split(
    data_dir: str,
    train_ratio: float = 0.8,
    min_ratings_per_user: int = 20,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    사용자별로 stratified split (각 사용자의 평점 일부를 test에)

    Args:
        data_dir: ml-20m 데이터 디렉토리
        train_ratio: 학습 데이터 비율
        min_ratings_per_user: 최소 평점 수 (필터링용)
        random_seed: 랜덤 시드

    Returns:
        train_df, test_df
    """
    import os

    ratings_path = os.path.join(data_dir, 'ratings.csv')
    print(f"평점 데이터 로드 중 (Stratified Split): {ratings_path}")

    ratings_df = pd.read_csv(ratings_path)
    print(f"  총 평점 수: {len(ratings_df):,}")

    # 평점이 충분한 사용자만 필터링
    user_counts = ratings_df['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
    print(f"  필터링 후 평점 수: {len(ratings_df):,}")
    print(f"  필터링 후 사용자 수: {ratings_df['userId'].nunique():,}")

    # 사용자별로 train/test 분할
    train_list = []
    test_list = []

    np.random.seed(random_seed)
    for user_id, group in ratings_df.groupby('userId'):
        n = len(group)
        n_train = int(n * train_ratio)

        # 랜덤 셔플
        group = group.sample(frac=1.0, random_state=random_seed + user_id)

        train_list.append(group.iloc[:n_train])
        test_list.append(group.iloc[n_train:])

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    print(f"  Train: {len(train_df):,} ({train_ratio*100:.1f}%)")
    print(f"  Test: {len(test_df):,} ({(1-train_ratio)*100:.1f}%)")

    return train_df, test_df