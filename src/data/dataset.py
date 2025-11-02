"""
영화 추천을 위한 데이터셋 클래스
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Any

class MovieRatingDataset(Dataset):
    """영화 제목과 평점 데이터셋"""
    
    def __init__(self, movie_data: pd.DataFrame, rating_path: str, vocab: Dict[str, int], max_length: int = 10):
        self.movie_data = movie_data
        self.rating_path = rating_path
        self.vocab = vocab
        self.max_length = max_length
        
        # 평점 데이터 전처리
        self.prepare_data()
    
    def prepare_data(self):
        """평점 데이터와 영화 데이터 결합"""
        # 평점 데이터 로드
        rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        self.ratings = pd.read_csv(self.rating_path, sep='\t', names=rating_columns)
        
        # 영화 정보와 결합
        self.data = self.ratings.merge(
            self.movie_data[['movie_id', 'clean_title', 'encoded_title']], 
            on='movie_id'
        )
        
        print(f"학습 데이터 크기: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        # 인코딩된 제목
        encoded_title = torch.tensor(row['encoded_title'], dtype=torch.long)
        
        # 평점 (1-5 -> 0-1 정규화)
        rating = torch.tensor((row['rating'] - 1) / 4.0, dtype=torch.float32)
        
        # 사용자 ID와 영화 ID (0부터 시작하도록 조정)
        user_id = torch.tensor(row['user_id'] - 1, dtype=torch.long)
        movie_id = torch.tensor(row['movie_id'] - 1, dtype=torch.long)
        
        return {
            'encoded_title': encoded_title,
            'rating': rating,
            'user_id': user_id,
            'movie_id': movie_id
        }