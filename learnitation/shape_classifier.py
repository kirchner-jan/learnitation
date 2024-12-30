import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_shape_probabilities(model, image_tensor):
    """Get log probabilities for each shape class"""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)
        log_probs = model(image_tensor)

    return log_probs.cpu().numpy()
