import torch
from torch import nn

from torch.utils.data import DataLoader
from dataset import ImageDataset
from model import NeuralNetwork, ConvAutoencoder

device = 'cuda' if torch.cuda.is_available() else'cpu'

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    model.to(device)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_loss = 0
    num_batches = len(dataloader)

    for batch, X in enumerate(dataloader):
        X = X.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, X)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
      
    print(f"Train loss : {train_loss/num_batches}")
    # À la fin de chaque epoch ou du training
    torch.cuda.empty_cache()

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.to(device)
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, X).item()

    test_loss /= num_batches
    return test_loss

def main():
    model = ConvAutoencoder()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    print("Importing data...")

    training_data = ImageDataset(r"..\data\bottle\bottle\train")

    batch_size = 32
    epochs = 50

    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        print("Done!")

if __name__ == "__main__":
    main()