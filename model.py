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
        
        return x.view(x.size(0), -1)

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(embeddings)
        out = self.linear(lstm_out)
        return out


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = CNNEncoder()
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    

    


