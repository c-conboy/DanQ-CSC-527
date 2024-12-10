from torch import nn

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
