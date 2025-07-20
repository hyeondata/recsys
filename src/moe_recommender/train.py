import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict

def calculate_ndcg(true_ratings, pred_ratings, k=10):
    """
    NDCG@k 계산 함수
    """
    # 예측값 기준으로 정렬된 인덱스
    pred_indices = np.argsort(pred_ratings)[::-1][:k]
    
    # 실제값 기준으로 정렬된 인덱스
    ideal_indices = np.argsort(true_ratings)[::-1][:k]
    
    # DCG 계산
    dcg = np.sum([true_ratings[idx] / np.log2(i + 2) for i, idx in enumerate(pred_indices)])
    
    # Ideal DCG 계산
    idcg = np.sum([true_ratings[idx] / np.log2(i + 2) for i, idx in enumerate(ideal_indices)])
    
    return dcg / idcg if idcg > 0 else 0

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Train", leave=False)
    for user, item, genre, rating in loop:
        user, item = user.to(device), item.to(device)
        genre, rating = genre.to(device), rating.to(device)
        optimizer.zero_grad()
        pred = model(user, item, genre)
        loss = criterion(pred, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = defaultdict(list)
    all_true_ratings = defaultdict(list)
    
    loop = tqdm(loader, desc="Eval", leave=False)
    for user, item, genre, rating in loop:
        user, item = user.to(device), item.to(device)
        genre, rating = genre.to(device), rating.to(device)
        pred = model(user, item, genre)
        loss = criterion(pred, rating)
        total_loss += loss.item()
        
        # 각 사용자별로 예측값과 실제값 저장
        for u, i, p, r in zip(user.cpu().numpy(), 
                            item.cpu().numpy(), 
                            pred.cpu().numpy(), 
                            rating.cpu().numpy()):
            all_predictions[u].append((i, p))
            all_true_ratings[u].append((i, r))
        
        loop.set_postfix(loss=loss.item())
    
    # NDCG@10 계산
    ndcg_scores = []
    for user in all_predictions.keys():
        # 현재 사용자의 모든 아이템에 대한 예측값과 실제값
        items_pred = all_predictions[user]
        items_true = dict(all_true_ratings[user])
        
        # 예측값 기준으로 정렬
        items_pred.sort(key=lambda x: x[1], reverse=True)
        
        # Top-10 아이템에 대한 실제 평점 가져오기
        top_k_items = items_pred[:10]
        true_ratings = np.array([items_true[item] for item, _ in top_k_items])
        pred_ratings = np.array([pred for _, pred in top_k_items])
        
        # 현재 사용자의 NDCG 계산
        ndcg = calculate_ndcg(true_ratings, pred_ratings, k=10)
        ndcg_scores.append(ndcg)
    
    avg_ndcg = np.mean(ndcg_scores)
    return total_loss / len(loader), avg_ndcg

def save_model(model, path='moe_model.pt'):
    torch.save(model.state_dict(), path)

def load_model(model, path='moe_model.pt'):
    model.load_state_dict(torch.load(path))
    return model