import torch
import torch.nn as nn

class HTRModel(nn.Module):
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes (int): number of output characters + 1 for CTC blank
        """
        super(HTRModel, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # (B,64,H,W)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B,64,H/2,W/2)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (B,128,H/2,W/2)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # (B,128,H/4,W/4)
        )

        # BiLSTM
        self.rnn = nn.LSTM(
            input_size=128 * 16,   # depends on normalized height (64 -> after pooling H=16)
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Fully connected (CTC output)
        self.fc = nn.Linear(512, num_classes)  # 256*2 (bidirectional)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, 1, H, W)
        Returns:
            out (Tensor): (B, W', num_classes)
        """
        conv = self.cnn(x)  # (B, C, H, W)
        b, c, h, w = conv.size()

        # Reshape for RNN: collapse H
        conv = conv.permute(0, 3, 1, 2)  # (B, W, C, H)
        conv = conv.contiguous().view(b, w, c * h)  # (B, W, features)

        # RNN
        rnn_out, _ = self.rnn(conv)  # (B, W, hidden*2)

        # Classification
        out = self.fc(rnn_out)  # (B, W, num_classes)
        return out
