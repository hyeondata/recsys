import pandas as pd
from sklearn.model_selection import train_test_split
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class MovieLensDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['UserID'].values, dtype=torch.long)
        self.movies = torch.tensor(df['MovieID'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['Rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


# Mixture of Experts (MoE) + Top-k + Dropout 모델 정의
class MoE_TopK_Dropout(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50, num_experts=4, top_k=2, expert_dropout=0.2):
        super(MoE_TopK_Dropout, self).__init__()
        self.num_experts = num_experts      # 전문가 수
        self.top_k = top_k                  # top-k 전문가 선택
        self.expert_dropout = expert_dropout  # 전문가 dropout 비율

        # 유저와 영화 임베딩 정의
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # Gating Network 정의 (user + movie embedding을 받아 expert score 출력)
        self.gate = nn.Linear(embedding_dim * 2, num_experts)

        # 여러 expert 네트워크 정의 (ModuleList 사용)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_experts)
        ])

    def forward(self, user_ids, movie_ids):
        # 임베딩 변환 및 결합
        user_embed = self.user_embedding(user_ids)      # [batch_size, embedding_dim]
        movie_embed = self.movie_embedding(movie_ids)     # [batch_size, embedding_dim]
        x = torch.cat([user_embed, movie_embed], dim=-1)    # [batch_size, embedding_dim*2]

        # Gating 네트워크로 전문가별 점수 계산
        gate_logits = self.gate(x)  # [batch_size, num_experts]

        # Top-k 전문가 선택
        topk_values, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_values, dim=-1)  # [batch_size, top_k]

        outputs = []
        for i in range(self.top_k):
            idx = topk_indices[:, i]       # [batch_size]
            weight = topk_weights[:, i]      # [batch_size]
            # expert dropout mask 적용
            mask = (torch.rand_like(weight) > self.expert_dropout).float()
            weight = weight * mask
            # 각 샘플에 대해 선택된 expert 실행
            expert_out = torch.stack([
                self.experts[expert_id](x[j].unsqueeze(0)).squeeze()
                for j, expert_id in enumerate(idx)
            ])
            outputs.append(expert_out * weight)
        # Top-k expert 결과 합산
        final_output = torch.stack(outputs, dim=0).sum(dim=0)  # [batch_size]
        return final_output

def main():

        # 파일 경로 설정 (다운로드한 폴더 기준)

    # ratings.dat 로드
    ratings = pd.read_csv(ratings_path, sep="::", engine='python',
                        names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    # movies.dat 로드
    movies = pd.read_csv(movies_path, sep="::", engine='python',
                        names=['MovieID', 'Title', 'Genres'])

    # 데이터 확인 (한번만 출력)
    print("Ratings Head:\n", ratings.head())
    print("Movies Head:\n", movies.head())

    # MovieLens 10M ratings 데이터 로드 및 인덱스 변환 (Embedding을 위해)
    ratings = pd.read_csv(ratings_path, sep="::", engine='python',
                        names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    user2idx = {id: idx for idx, id in enumerate(ratings['UserID'].unique())}
    movie2idx = {id: idx for idx, id in enumerate(ratings['MovieID'].unique())}
    ratings['UserID'] = ratings['UserID'].map(user2idx)
    ratings['MovieID'] = ratings['MovieID'].map(movie2idx)

    # Train/Test Split
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    print(len(test_data))
    # PyTorch Dataset 정의


    train_dataset = MovieLensDataset(train_data)
    test_dataset = MovieLensDataset(test_data)

    num_cores = multiprocessing.cpu_count()
    print(f"Available CPU cores: {num_cores}")

    # DataLoader 설정 (배치 크기가 크므로 num_workers는 CPU 코어 수 사용)
    train_loader = DataLoader(train_dataset, batch_size=32768, shuffle=True, num_workers=num_cores)
    test_loader = DataLoader(test_dataset, batch_size=32768, shuffle=False, num_workers=num_cores)
    # 디바이스 설정 (MPS 우선, 없으면 CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 초기화 (유저 수, 영화 수를 인자로 넘김)
    num_users = len(user2idx)
    num_movies = len(movie2idx)
    model = MoE_TopK_Dropout(num_users, num_movies, num_experts=4, top_k=2, expert_dropout=0.2).to(device)

    # 손실 함수 및 최적화 함수 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        # tqdm을 사용해 학습 진행도 표시 (진행바는 학습 루프 내에서 업데이트)
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)
        for user_ids, movie_ids, ratings in train_bar:
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
            optimizer.zero_grad()
            outputs = model(user_ids, movie_ids)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Time: {elapsed:.2f}s")
    
    # 평가 단계
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing", leave=False)
        for user_ids, movie_ids, ratings in test_bar:
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
            outputs = model(user_ids, movie_ids)
            loss = criterion(outputs, ratings)
            total_loss += loss.item()
            test_bar.set_postfix(loss=f"{loss.item():.4f}")
    print(f"Test Loss (MSE): {total_loss/len(test_loader):.4f}")

if __name__ == "__main__":
    print("Starting training...")
    main()

