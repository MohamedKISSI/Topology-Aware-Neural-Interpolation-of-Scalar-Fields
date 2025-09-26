import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import set_seed

set_seed(42)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + residual)


# Positional Encoding for time
class PositionalEncoding_512(nn.Module):
    def __init__(self, embedding_dim=128, max_time_steps=200, omega_0=2*torch.pi):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_time_steps = max_time_steps
        self.omega_0 = omega_0

    def forward(self, t):

        t_norm = t.float().clamp(0, self.max_time_steps - 1) / self.max_time_steps
        position = t_norm.unsqueeze(-1)

        div_term = torch.exp(
            torch.arange(0, self.embedding_dim // 2, dtype=torch.float, device=t.device)
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pe = torch.zeros(t.shape[0], self.embedding_dim, device=t.device)
        pe[:, 0::2] = torch.sin(self.omega_0 * position * div_term)
        pe[:, 1::2] = torch.cos(self.omega_0 * position * div_term)

        return pe


class TimeToImageGenerator_512(nn.Module):
    def __init__(self, time_embedding_dim=128, max_time_steps=200):
        super(TimeToImageGenerator_512, self).__init__()
        self.positional_encoding = PositionalEncoding_512(embedding_dim=time_embedding_dim, max_time_steps=max_time_steps)

        self.time_projection = nn.Sequential(
            nn.Linear(time_embedding_dim, 256 * 8 * 8),
            nn.ReLU(),
        )

        self.cnn_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8x8 - 16x16
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            ResBlock(128),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16x16 - 32x32
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            ResBlock(64),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32x32 - 64x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            ResBlock(64),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x64 - 128x128
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            ResBlock(32),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128x128 - 256x256
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            ResBlock(16),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 256x256 - 512x512
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            ResBlock(8),

            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, t):
        time_embedding = self.positional_encoding(t)
        latent_space = self.time_projection(time_embedding).view(-1, 256, 8, 8)
        generated_image = self.cnn_decoder(latent_space)
        return generated_image

