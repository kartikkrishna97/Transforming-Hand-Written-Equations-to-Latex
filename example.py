import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
    
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.pool6 = nn.AvgPool2d(kernel_size=3)  
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        print(x.shape)
        x = self.pool2(self.relu2(self.conv2(x)))
        print(x.shape)
        x = self.pool3(self.relu3(self.conv3(x)))
        print(x.shape)
        x = self.pool4(self.relu4(self.conv4(x)))
        print(x.shape)
        x = self.pool6(self.pool5(self.relu5(self.conv5(x))))
        print(x.shape)
        
        return x

# Initialize the CNN encoder
encoder = CNNEncoder()

# Example input image dimensions (1 channel, 224x224)
input_image = torch.randn(1, 1, 224, 224)

# Forward pass to get the encoded features
encoded_features = encoder(input_image)

# Check the shape of the encoded features
# print(encoded_features.shape)
