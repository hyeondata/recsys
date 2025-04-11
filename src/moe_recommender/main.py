import torch
from torch import nn, optim
from data_loader import load_all_data, get_loaders
from models.moe_model import MoEModel
from train import train, evaluate, save_model, load_model

from dotenv import load_dotenv
import os


def main(mode='train'):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    ratings_path = os.getenv("RATINGS_PATH")
    movies_path = os.getenv("MOVIES_PATH")

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
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Test MSE: {test_loss:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    args = parser.parse_args()
    main(mode=args.mode)