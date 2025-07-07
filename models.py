import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights

# Note:
# All models output raw logits of shape [B, num_classes, H, W].
# These can be used with either one-hot encoded labels or
# class index labels, depending on loss function and label preprocessing.

##### DEEPLAB #####
class DeepLabV3(nn.Module):
    
    def __init__(self, num_channels=128, num_classes=9, pretrained=True):
        super(DeepLabV3, self).__init__()
        
        # backbone
        self.model = deeplabv3_resnet50(pretrained=pretrained)
        
        # classifier head
        self.model.classifier = nn.Sequential(
            nn.Conv2d(2048, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(num_channels, num_classes, kernel_size = 1)
        )
    
    def forward(self, x):
        return self.model(x)['out']

##### DEEPLABLITE #####

class DeepLabV3Lite(nn.Module):
    def __init__(self, num_channels=64, num_classes=9):
        super().__init__()
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.model = deeplabv3_mobilenet_v3_large(weights=weights)
        
        self.model.classifier = nn.Sequential(
            nn.Conv2d(960, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(num_channels, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        return self.model(x)['out']

class DeepLabV3LiteV2(nn.Module):
    def __init__(self, num_channels=64, num_classes=9, pretrained=True):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(pretrained=pretrained)
        
        self.model.classifier = nn.Sequential(
            nn.Conv2d(960, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        return self.model(x)['out']


#### BASIC CNN ####

class CNN_7_Layers(nn.Module): 
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

##### UNET #####


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNetLite(nn.Module):
    def __init__(self, in_channels=3, num_classes=9, base_c=32):
        super().__init__()
        c = base_c

        #encoder
        self.enc1 = ConvBlock(in_channels, c)     # [B, c, H, W]
        self.enc2 = ConvBlock(c, c*2)             # [B, 2c, H/2, W/2]
        self.enc3 = ConvBlock(c*2, c*4)           # [B, 4c, H/4, W/4]
        self.enc4 = ConvBlock(c*4, c*8)           # [B, 8c, H/8, W/8]

        self.pool = nn.MaxPool2d(2)

        #bottleneck
        self.bottleneck = ConvBlock(c*8, c*16)    # [B, 16c, H/16, W/16]

        #decoder
        self.up4 = nn.ConvTranspose2d(c*16, c*8, 2, stride=2)
        self.dec4 = ConvBlock(c*16, c*8)

        self.up3 = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.dec3 = ConvBlock(c*8, c*4)

        self.up2 = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec2 = ConvBlock(c*4, c*2)

        self.up1 = nn.ConvTranspose2d(c*2, c, 2, stride=2)
        self.dec1 = ConvBlock(c*2, c)

 	#output
        self.out_conv = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
  	#encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

	#bottleneck
        b = self.bottleneck(self.pool(e4))

        #decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # shape: [B, num_classes, H, W]
