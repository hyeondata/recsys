import torch
from torch import nn, optim
from data_loader import load_all_data, get_loaders
from models.moe_model import MoEModel
from train import train, evaluate, save_model, load_model
from torchinfo import summary

from dotenv import load_dotenv
import os

def display_model_info(model, train_loader, device):
    """모델의 구조와 파라미터 정보를 출력하는 함수"""
    batch = next(iter(train_loader))
    batch_size = batch[0].shape[0]
    movie_feat_dim = batch[2].shape[1]
    
    print("\n=== Model Architecture ===")
    summary(model, 
            input_size=[(batch_size,), (batch_size,), (batch_size, movie_feat_dim)],
            dtypes=[torch.int64, torch.int64, torch.float32],
            device=device)
    print("\n=== Model Parameters ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("=====================\n")

def main(mode='train'):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    load_dotenv()
    ratings_path = os.getenv("RATINGS_PATH")
    movies_path = os.getenv("MOVIES_PATH")
    print(f"Ratings path: {ratings_path}, Movies path: {movies_path}")
    train_ds, test_ds, num_users, num_items, movie_feat_dim = load_all_data(ratings_path, movies_path)
    print(f"Number of users: {num_users}, Number of items: {num_items}, Movie feature dimension: {movie_feat_dim}")
    train_loader, test_loader = get_loaders(train_ds, test_ds)

    model = MoEModel(num_users, num_items, movie_feat_dim, embed_dim=32, num_experts=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    if mode == 'train':
        for epoch in range(10):
            loss = train(model, train_loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1} | Loss: {loss:.4f}")
        save_model(model)

    elif mode == 'test':
        load_model(model)
        test_loss, ndcg = evaluate(model, test_loader, criterion, device)
        print(f"Test MSE: {test_loss:.4f}")
        print(f"NDCG@10: {ndcg:.4f}")

    elif mode == 'param':
        # Display model information
        display_model_info(model, train_loader, device)
        print("Model parameters displayed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'param'], default='param')
    args = parser.parse_args()
    main(mode=args.mode)