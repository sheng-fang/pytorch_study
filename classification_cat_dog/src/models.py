import torch
import torch.nn as nn
import torchvision


class FCHead(nn.Module):
    def __init__(self, layer_cfg):
        """

        Args:
            layer_cfg: a list of integers, which indicates the hidden nodes for
            each layer
        """
        super().__init__()
        self.layer_cfg = layer_cfg
        modules = []
        assert len(layer_cfg) >= 2, "needs information for at least 2 layers"
        for idx in range(len(layer_cfg) - 1):
            modules.append(
                nn.Linear(layer_cfg[idx], layer_cfg[idx + 1]))
            if idx < len(layer_cfg) - 2:
                modules.append(nn.ReLU(inplace=True))
                modules.append(nn.Dropout(p=0.5))

        # modules.append(nn.Softmax(dim=1))

        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        x = self.fc(x)

        return x


class PassHead(nn.Module):
    def __init__(self):
        """

        Args:
            layer_cfg: a list of integers, which indicates the hidden nodes for
            each layer
        """
        super().__init__()

    def forward(self, x):

        return x
