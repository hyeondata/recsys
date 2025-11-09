"""
ml-20m 영화 데이터 전처리 모듈
- CSV 형식 (movieId, title, genres)
- 장르는 파이프로 구분된 문자열 형식
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

class MoviePreprocessor:
    """ml-20m 영화 데이터 전처리 클래스"""

    def __init__(self, data_dir: str):
        import os
        if os.path.isdir(data_dir):
            self.movie_path = os.path.join(data_dir, 'movies.csv')
        else:
            self.movie_path = data_dir
        self.vocab = None

    def load_movie_data(self) -> pd.DataFrame:
        """영화 데이터 로드 (CSV 형식)"""
        movie_df = pd.read_csv(self.movie_path)
        # movieId, title, genres 컬럼 있음
        return movie_df

    def parse_genres(self, genres_str: str) -> List[str]:
        """파이프로 구분된 장르 문자열 파싱"""
        if pd.isna(genres_str) or genres_str == '(no genres listed)':
            return []
        return genres_str.split('|')
    
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
        print("영화 데이터 전처리 시작 (ml-20m)...")

        # 1. 데이터 로드
        movie_df = self.load_movie_data()
        print(f"  로드된 영화 수: {len(movie_df)}")

        # 2. 장르 파싱
        movie_df['genre_list'] = movie_df['genres'].apply(self.parse_genres)

        # 3. 제목 전처리
        processed_titles = movie_df['title'].apply(self.preprocess_title)
        movie_df['clean_title'] = [x[0] for x in processed_titles]
        movie_df['year'] = [x[1] for x in processed_titles]

        # 4. 단어 사전 구축
        vocab = self.build_vocab(movie_df['clean_title'].values)
        print(f"  구축된 사전 크기: {len(vocab)}")

        # 5. 제목 인코딩
        encoded_titles = self.encode_titles(movie_df['clean_title'].values, vocab)
        movie_df['encoded_title'] = [encoded_titles[i].tolist() for i in range(len(encoded_titles))]

        print("영화 데이터 전처리 완료!")
        return movie_df, vocab, encoded_titles

    def get_genre_embedding_dict(self, movie_df: pd.DataFrame) -> Dict[int, List[str]]:
        """영화 ID별 장르 리스트 반환"""
        return dict(zip(movie_df['movieId'], movie_df['genre_list']))