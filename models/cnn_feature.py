import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class CNNFeature(nn.Module):
    def __init__(self, ntoken):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, ntoken)

    def forward(self, images):
        # images: batch_size x channel x height x width
        images = F.relu(F.max_pool2d(self.conv1(images), 2))
        images = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(images)), 2))
        batch_size = images.shape[0]
        # seq_len = 48
        images = images.permute(0, 3, 1, 2).reshape(batch_size, 48, -1)
        images = images.permute(1, 0, 2)
        images = self.fc1(images)
        images = F.dropout(images, training=self.training)
        images = self.fc2(images)
        images = F.log_softmax(images, dim=2)
        return images
