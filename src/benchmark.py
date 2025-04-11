import torch
import time
import psutil
import os
from torch.utils.data import DataLoader
import multiprocessing
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

def benchmark_batch_sizes(dataset, model_class, loss_fn, optimizer_class, batch_sizes, device=None, num_workers=0, epochs=1):
    """
    다양한 배치 크기에 대한 학습 시간을 벤치마킹하는 함수.

    매개변수:
    - dataset (torch.utils.data.Dataset): 학습에 사용할 데이터셋.
    - model_class (torch.nn.Module): 모델 클래스.
    - loss_fn (torch.nn.Module): 손실 함수.
    - optimizer_class (torch.optim.Optimizer): 옵티마이저 클래스.
    - batch_sizes (list of int): 테스트할 배치 크기들의 리스트.
    - device (torch.device, optional): 사용할 디바이스. 기본값은 자동 감지.
    - num_workers (int, optional): DataLoader에 사용할 워커 수. 기본값은 0.
    - epochs (int, optional): 각 배치 크기에 대해 학습할 에폭 수. 기본값은 1.

    반환값:
    - results (list of tuples): 각 배치 크기에 대한 (배치 크기, 평균 시간, 평균 메모리 사용량)의 리스트.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    if num_workers == 0:
        num_workers = multiprocessing.cpu_count() // 2

    print(f"Available CPU cores: {num_workers}")

    def get_memory_usage():
        """현재 프로세스의 메모리 사용량을 MB 단위로 반환."""
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 ** 2  # MB 단위
        return mem

    results = []

    for batch_size in batch_sizes:
        print(f"\nTrying batch size: {batch_size}")
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            model = model_class().to(device)
            optimizer = optimizer_class(model.parameters())

            model.train()
            total_time = 0.0
            total_mem = 0.0

            for epoch in range(epochs):
                start_time = time.time()
                mem_before = get_memory_usage()

                for batch in dataloader:
                    # batch는 튜플 형태로, 데이터셋의 반환 값들을 포함합니다.
                    # 필요한 만큼의 변수를 선언하여 언패킹합니다.
                    if len(batch) == 2:
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                    elif len(batch) == 3:
                        users, items, ratings = batch
                        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                        outputs = model(users, items)
                        loss = loss_fn(outputs, ratings)
                    else:
                        raise ValueError(f"Unexpected number of elements in batch: {len(batch)}")
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if device.type == 'cuda' or device.type == 'mps':
                    torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()

                mem_after = get_memory_usage()
                end_time = time.time()

                elapsed = end_time - start_time
                mem_used = mem_after - mem_before

                total_time += elapsed
                total_mem += mem_used

            avg_time = total_time / epochs
            avg_mem = total_mem / epochs

            print(f"Batch size {batch_size}: Avg Time = {avg_time:.2f}s | Avg Memory Used = {avg_mem:.2f} MB")
            results.append((batch_size, avg_time, avg_mem))

        except RuntimeError as e:
            print(f"Batch size {batch_size} failed: {e}")

    return results




###



# ratings.dat 로드
ratings = pd.read_csv(ratings_path, sep="::", engine='python',
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# movies.dat 로드
movies = pd.read_csv(movies_path, sep="::", engine='python',
                     names=['MovieID', 'Title', 'Genres'])

# ID를 0부터 시작하는 인덱스로 변환 (Embedding을 위해)
user2idx = {id: idx for idx, id in enumerate(ratings['UserID'].unique())}
movie2idx = {id: idx for idx, id in enumerate(ratings['MovieID'].unique())}

ratings['UserID'] = ratings['UserID'].map(user2idx)
ratings['MovieID'] = ratings['MovieID'].map(movie2idx)

# Train/Test Split
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# PyTorch Dataset 클래스 정의
class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['UserID'].values, dtype=torch.long)
        self.movies = torch.tensor(df['MovieID'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['Rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# Dataset 인스턴스 생성
train_dataset = MovieLensDataset(train_data)
test_dataset = MovieLensDataset(test_data)

# Matrix Factorization 모델 정의
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, emb_size=50):
        super(MatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_emb(user_ids)
        item_vecs = self.item_emb(item_ids)
        return (user_vecs * item_vecs).sum(1)


if __name__ == "__main__":
# 벤치마킹 실행
    num_users = len(user2idx)
    num_items = len(movie2idx)

    # 모델 클래스 정의를 람다 함수로 감싸서 전달
    model_class = lambda: MatrixFactorization(num_users, num_items)

    loss_function = nn.MSELoss()
    optimizer_class = lambda params: optim.Adam(params, lr=0.005)
    batch_sizes_to_test = [65536, 131072, 262144, 524288, 1048576]

    benchmark_results = benchmark_batch_sizes(
        dataset=train_dataset,
        model_class=model_class,
        loss_fn=loss_function,
        optimizer_class=optimizer_class,
        batch_sizes=batch_sizes_to_test,
        epochs=1
    )

    # 결과 출력
    print("\n=== Benchmark Results ===")
    for batch_size, avg_time, avg_mem in benchmark_results:
        print(f"Batch Size: {batch_size} | Avg Time: {avg_time:.2f}s | Avg Memory Used: {avg_mem:.2f} MB")