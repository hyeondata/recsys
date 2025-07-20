import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import multiprocessing

class MovieLensDataset(Dataset):
    def __init__(self, df, movie_features):
        self.users = torch.tensor(df['user'].values, dtype=torch.long)
        self.items = torch.tensor(df['item'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['Rating'].values, dtype=torch.float)
        self.movie_features = torch.tensor(movie_features.values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        rating = self.ratings[idx]
        genre_feat = self.movie_features[item]
        return user, item, genre_feat, rating


def load_all_data(ratings_path, movies_path):


    # ratings.dat 로드
    ratings = pd.read_csv(ratings_path, sep="::", engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

    # movies.dat 로드
    movies = pd.read_csv(movies_path, sep="::", engine='python',
                         names=['MovieID', 'Title', 'Genres'])
    # 인코딩
    user_encoder = LabelEncoder()
    # item_encoder = LabelEncoder()

    all_movie_ids = pd.concat([ratings["MovieID"], movies["MovieID"]])
    item_encoder = LabelEncoder()
    item_encoder.fit(all_movie_ids)


    ratings["user"] = user_encoder.fit_transform(ratings["UserID"])
    ratings["item"] = item_encoder.transform(ratings["MovieID"])

    # 영화 장르 → 다중 핫 인코딩
    genre_set = set()
    for g in movies["Genres"]:
        genre_set.update(g.split("|"))

    for genre in genre_set:
        movies[genre] = movies["Genres"].apply(lambda x: int(genre in x.split("|")))

    movies["item"] = item_encoder.transform(movies["MovieID"])
    movie_features = movies.set_index("item")[sorted(genre_set)].sort_index()

    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

    return (
        MovieLensDataset(train_df, movie_features),
        MovieLensDataset(test_df, movie_features),
        ratings["user"].nunique(),
        ratings["item"].nunique(),
        movie_features.shape[1],
    )

def get_loaders(train_ds, test_ds, batch_size=8192):
    # 최적의 워커 수 계산 - 코어당 하나의 워커 권장
    num_workers = max(4, multiprocessing.cpu_count() - 1)
    
    # 메모리와 성능 최적화를 위한 DataLoader 설정
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # 텐서를 CUDA 고정 메모리에 배치하여 호스트-GPU 전송 속도 향상
        prefetch_factor=2,  # 미리 가져올 배치 수 설정
        persistent_workers=True,  # 워커 프로세스 재사용하여 시작 비용 절감
        drop_last=True  # 마지막 불완전한 배치 제거하여 일관된 배치 크기 보장
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size,
        shuffle=False,  # 테스트 데이터는 셔플 필요 없음
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader