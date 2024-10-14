import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from model import ViT_LSTM

# Set the device to MPS (for Apple silicon). If not available, fallback to CPU
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# Instantiate the model
model = ViT_LSTM().to(device)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()

        for i, (videos, labels) in enumerate(train_loader):
            # Move data to the target device (MPS or CPU)
            videos = videos.to(device)
            labels = labels.float().to(device)  # Labels need to be float for BCELoss

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(videos).squeeze()  # Get model outputs
            loss = criterion(outputs, labels)  # Compute the loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track the running loss
            running_loss += loss.item()

            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}] completed in {epoch_duration:.2f} seconds. Loss: {running_loss / len(train_loader):.4f}')

    print('Training complete!')

# Example usage
if __name__ == "__main__":
    # Assuming the DataLoader is already defined
    train_dataset = VideoDataset(root_dir='/path/to/video/root', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)
