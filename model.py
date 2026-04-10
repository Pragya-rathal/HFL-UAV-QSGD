### model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, image_size=28):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Compute FC input size dynamically
        dummy = torch.zeros(1, in_channels, image_size, image_size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        fc_input = dummy.view(1, -1).shape[1]

        self.fc1 = nn.Linear(fc_input, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_model(config):
    model = CNN(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        image_size=config['image_size']
    )
    return model


def flatten_model(model):
    """Flatten all model parameters into a single 1D tensor."""
    params = [p.data.view(-1) for p in model.parameters()]
    return torch.cat(params)


def load_model(model, flat_tensor):
    """Load a flattened 1D tensor back into model parameters."""
    idx = 0
    for p in model.parameters():
        num = p.numel()
        p.data.copy_(flat_tensor[idx:idx + num].view(p.shape))
        idx += num


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
