import torch
from torch import nn


def activation_layer(name: str, elu_alpha: float = 1.0) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU(alpha=elu_alpha)
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unknown activation: {name}")


class EEGNet(nn.Module):

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 2,
        n_samples: int = 750,
        activation: str = "elu",
        elu_alpha: float = 1.0,
        dropout: float = 0.25,
    ):
        super().__init__()
        if n_channels != 2 or n_samples != 750:
            raise ValueError("This Lab2 implementation expects 2 EEG channels x 750 samples.")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0,1).")
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            activation_layer(activation, elu_alpha),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout),
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            activation_layer(activation, elu_alpha),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout),
        )
        self.classify = nn.Sequential(nn.Flatten(), nn.Linear(32 * 23, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return self.classify(x)


class DeepConvNet(nn.Module):

    def __init__(self, n_classes: int = 2, n_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        if n_channels != 2:
            raise ValueError("DeepConvNet is configured for 2 EEG channels.")
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=False),
            nn.Conv2d(25, 25, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),
        )
        self.block2 = self._block(25, 50, dropout)
        self.block3 = self._block(50, 100, dropout)
        self.block4 = self._block(100, 200, dropout)
        self.classify = nn.Sequential(nn.Flatten(), nn.Linear(200 * 43, n_classes))

    @staticmethod
    def _block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classify(x)
