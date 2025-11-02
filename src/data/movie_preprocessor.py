"""
영화 데이터 전처리 모듈
- 제목 전처리
- 단어 사전 구축
- 텍스트 인코딩
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

class MoviePreprocessor:
    """영화 데이터 전처리 클래스"""
    
    def __init__(self, data_dir: str, genre_path: Optional[str] = None):
        # data_dir이 디렉토리면 u.item을 찾고, 파일이면 그대로 사용
        import os
        if os.path.isdir(data_dir):
            self.movie_path = os.path.join(data_dir, 'u.item')
            if genre_path is None:
                genre_path = os.path.join(data_dir, 'u.genre')
        else:
            self.movie_path = data_dir
        self.genre_path = genre_path
        self.vocab = None
        self.genre_columns = [
            'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
            'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western'
        ]
        
    def load_movie_data(self) -> pd.DataFrame:
        """영화 데이터 로드"""
        columns = [
            'movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'
        ] + self.genre_columns
        
        movie_df = pd.read_csv(self.movie_path, sep="|", names=columns, encoding='latin1')
        movie_df.drop(['release_date', 'video_release_date', 'IMDb_URL'], axis=1, inplace=True)
        
        return movie_df
    
    def preprocess_title(self, title: str) -> Tuple[str, Optional[int]]:
        """영화 제목 전처리"""
        # 년도 추출
        year_match = re.search(r'\((\d{4})\)', title)
        year = int(year_match.group(1)) if year_match else None
        
        # 년도와 괄호 제거
        clean_title = re.sub(r'\s*\(\d{4}\)\s*', '', title)
        
        # 특수문자와 불필요한 공백 정리
        clean_title = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_title)
        clean_title = ' '.join(clean_title.split())  # 연속 공백 제거
        
        return clean_title.lower().strip(), year
    
    def build_vocab(self, titles: List[str], min_freq: int = 1) -> Dict[str, int]:
        """제목들로부터 단어 사전 구축"""
        word_counts = {}
        for title in titles:
            words = title.split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # 최소 빈도 이상인 단어들만 포함
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in word_counts.items():
            if count >= min_freq:
                vocab[word] = len(vocab)
        
        self.vocab = vocab
        return vocab
    
    def encode_titles(self, titles: List[str], vocab: Dict[str, int], max_length: int = 10) -> np.ndarray:
        """제목들을 숫자 시퀀스로 변환"""
        encoded = []
        for title in titles:
            words = title.split()[:max_length]  # 최대 길이 제한
            encoded_title = [vocab.get(word, vocab['<UNK>']) for word in words]
            
            # 패딩
            while len(encoded_title) < max_length:
                encoded_title.append(vocab['<PAD>'])
            
            encoded.append(encoded_title)
        
        return np.array(encoded)
    
    def process_movie_data(self) -> Tuple[pd.DataFrame, Dict[str, int], np.ndarray]:
        """전체 영화 데이터 전처리 파이프라인"""
        print("영화 데이터 전처리 시작...")
        
        # 1. 데이터 로드
        movie_df = self.load_movie_data()
        print(f"  로드된 영화 수: {len(movie_df)}")
        
        # 2. 제목 전처리
        processed_titles = movie_df['title'].apply(self.preprocess_title)
        movie_df['clean_title'] = [x[0] for x in processed_titles]
        movie_df['year'] = [x[1] for x in processed_titles]
        
        # 3. 단어 사전 구축
        vocab = self.build_vocab(movie_df['clean_title'].values)
        print(f"  구축된 사전 크기: {len(vocab)}")
        
        # 4. 제목 인코딩
        encoded_titles = self.encode_titles(movie_df['clean_title'].values, vocab)
        movie_df['encoded_title'] = [encoded_titles[i].tolist() for i in range(len(encoded_titles))]
        
        print("영화 데이터 전처리 완료!")
        return movie_df, vocab, encoded_titles
    
    def get_genre_features(self, movie_df: pd.DataFrame) -> np.ndarray:
        """장르 one-hot 특성 추출"""
        return movie_df[self.genre_columns].values.astype(np.float32)