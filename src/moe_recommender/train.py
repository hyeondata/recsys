import torch
from tqdm import tqdm

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
    loop = tqdm(loader, desc="Eval", leave=False)
    for user, item, genre, rating in loop:
        user, item = user.to(device), item.to(device)
        genre, rating = genre.to(device), rating.to(device)
        pred = model(user, item, genre)
        loss = criterion(pred, rating)
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def save_model(model, path='moe_model.pt'):
    torch.save(model.state_dict(), path)

def load_model(model, path='moe_model.pt'):
    model.load_state_dict(torch.load(path))
    return model