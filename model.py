import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Segmentation(nn.Module): 
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # downsampling
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # [B, 32, 544, 1024]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 272, 512]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 272, 512]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 136, 256]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [B, 128, 136, 256]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 128, 68, 128]

            #upsampling
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # [B, 64, 136, 256]
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # [B, 32, 272, 512]
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),    # [B, 16, 544, 1024]
            nn.ReLU(),

            nn.Conv2d(16, 9, kernel_size=1),  # final class logits: [B, 9, 544, 1024]
        )


    def forward(self, x): # [B, 3, 544, 1024]
        return self.layers(x)  # [B, 9, 544, 1024]
