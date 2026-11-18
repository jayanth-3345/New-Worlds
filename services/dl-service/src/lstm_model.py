
import torch
import torch.nn as nn


class TransitLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Convert [batch, channels, sequence]
        # into [batch, sequence, channels]
        x = x.transpose(1, 2)

        _, (hidden, _) = self.lstm(x)

        final_hidden = hidden[-1]

        return self.classifier(final_hidden)


if __name__ == "__main__":
    model = TransitLSTM()
    sample = torch.randn(4, 1, 201)
    output = model(sample)

    print("Model:", model.__class__.__name__)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
