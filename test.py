import torch
import scipy
import numpy as np
from torch import nn
import h5py

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
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
print("Done")

# Defining Model
print("Loading Model...")
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
    

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X) 
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

model = DANQ().to(device)
model.load_state_dict(torch.load("./best_model_test_data.pth", weights_only=True))
loss_fn = nn.BCELoss()
test(test_dataloader, model, loss_fn)