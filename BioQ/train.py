import torch
import scipy
import numpy as np
from torch import nn

# Setting Device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Loading Data
print("Loading Data...")
mat = scipy.io.loadmat('./data/test.mat')
data = mat['testxdata']
target = mat['testdata']
data = torch.from_numpy(data).to(dtype=torch.float32)
target = torch.from_numpy(target).to(dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(data, target)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
print("Done")

# Defining Model
print("Loading Model...")
class DANQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = 4, out_channels = 320, kernel_size = 26, padding='valid', stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(p=0.2),
        )
        self.brnn = nn.Sequential(
            nn.LSTM(input_size=320, hidden_size=320, batch_first=True, bidirectional=True, num_layers=2),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(in_features = 75*640, out_features = 925),
            nn.ReLU(),
            nn.Linear(in_features = 925, out_features = 919),
            nn.Sigmoid()
        )

    def forward(self, x):
        convolutions = self.conv(x)  # Conv1d + ReLU + MaxPool1d + Dropout
        convolutions = convolutions.transpose(1, 2)  # Shape: (batch_size, sequence_length, 320)
        outputs, _ = self.brnn[0](convolutions)  # LSTM layer
        outputs = self.brnn[1](outputs)  # Apply dropout after LSTM
        outputs = self.brnn[2](outputs)  # Flatten
        outputs = self.brnn[3](outputs)  # Linear layer
        outputs = self.brnn[4](outputs)  # ReLU
        outputs = self.brnn[5](outputs)  # Final linear layer
        outputs = self.brnn[6](outputs)  # Sigmoid activation

        return outputs


model = DANQ().to(device)
print(model)
print("Done")


# Training
loss_fn = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

size = len(train_dataloader.dataset)
model.train()

for epoch in range(50):
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")