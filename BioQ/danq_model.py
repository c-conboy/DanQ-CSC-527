from torch import nn

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
