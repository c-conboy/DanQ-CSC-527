# Import packages
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import h5py
import numpy as np
import matplotlib.pyplot as plt

#Dataset Class
class HDF5Dataset(Dataset):
    def __init__(self, file_path, data_key, target_key):
        self.file_path = file_path
        self.data_key = data_key
        self.target_key = target_key
        self.h5_file = h5py.File(file_path, "r")  
        self.data = self.h5_file[data_key]
        self.target = self.h5_file[target_key]
        self.num_samples = self.data.shape[2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = torch.tensor(self.data[:, :, idx], dtype=torch.float32).permute(1,0)
        target = torch.tensor(self.target[:, idx], dtype=torch.float32)
        return data, target

    def __del__(self):
        self.h5_file.close() 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Dataset and DataLoader
file_path = "./data/train.mat"
data_key = "trainxdata"
target_key = "traindata"

dataset = HDF5Dataset(file_path, data_key, target_key)
train_dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

# Model definition
class DANQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(p=0.2),
        )
        self.brnn = nn.Sequential(
            nn.LSTM(input_size=320, hidden_size=320, num_layers=2, batch_first=True, bidirectional=True),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(in_features=75 * 640, out_features=925),
            nn.ReLU(),
            nn.Linear(in_features=925, out_features=919),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        outputs, _ = self.brnn[0](x)
        outputs = self.brnn[1:](outputs)
        return outputs

# Training function
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        epoch_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    return epoch_loss / len(dataloader)

model = DANQ().to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5, alpha=0.9)

epochs = 500
best_loss = float("inf")
loss_history = []

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n{'-' * 30}")
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    loss_history.append(train_loss)
    print(f"Epoch {epoch + 1} Loss: {train_loss:.6f}")

    # Update plot
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Loss" if epoch == 0 else "", color="blue")
    plt.legend()
    plt.savefig("training_loss.png") 
    plt.clf()

    # Save the best model
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Saved Best Model with Loss: {best_loss:.6f}")

print("Training Complete!")
