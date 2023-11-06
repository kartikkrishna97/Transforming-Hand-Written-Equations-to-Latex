import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
import glob
import pandas as pd
import cv2
import os 
from preprocess import preprocess_text
import torch
import torch.nn as nn
from model import ImageCaptioningModel
import torch.optim as optim



image_train_path = 'col_774_A4_2023/HandwrittenData/images/train/'
text_train_path = 'col_774_A4_2023/HandwrittenData/train_hw.csv'
text_val_path = 'col_774_A4_2023/HandwrittenData/val_hw.csv'

images_train = glob.glob(image_train_path+'*png')


image_train_new = []
for i in range(len(images_train)):
    final_images = images_train[i].split('/')
    image_train_new.append(final_images[-1])


X_train, word2index1, index2word2 = preprocess_text(text_train_path)
X_val, word2index2, index2word2 = preprocess_text(text_val_path)


class NormalizeImage:
    def __call__(self, image):
        min_value = image.min()
        max_value = image.max()

        normalized_image = (image - min_value) / (max_value - min_value)

        return normalized_image


class TrainImageTextDataset(Dataset):
    def __init__(self, text_file_path, transform) -> None:
        super().__init__()
        self.text_data = pd.read_csv(text_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        image_name_idx = self.text_data['image'][idx]
        text = X_train[idx]
        image_folder = os.path.join(image_train_path+image_name_idx)
        image = Image.open(image_folder)
        if self.transform:
            image = self.transform(image)

        # text = self.text_data.loc[self.text_data['image'] == ', 'formula'].values[0]


        return image, text
    
class ValImageTextDataset(Dataset):
    def __init__(self, text_file_path, transform) -> None:
        super().__init__()
        self.text_data = pd.read_csv(text_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        image_name_idx = self.text_data['image'][idx]
        text = X_train[idx]
        image_folder = os.path.join(image_train_path+image_name_idx)
        image = Image.open(image_folder)
        if self.transform:
            image = self.transform(image)

        # text = self.text_data.loc[self.text_data['image'] == ', 'formula'].values[0]


        return image, text


def custom_collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    max_batch_text_length = max(len(text) for text in texts)
    padded_texts = []

    for text in texts:
        padding_length = max_batch_text_length - len(text)
        padded_text = text + [2] * padding_length
        padded_texts.append(padded_text)


    padded_texts = torch.tensor(padded_texts)
    return images, padded_texts


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NormalizeImage()
])


train_image_text_dataset = TrainImageTextDataset(text_train_path, transform=image_transform)
val_image_text_dataset = ValImageTextDataset(text_val_path, transform=image_transform)

train_dataloader = DataLoader(train_image_text_dataset, batch_size=32, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_image_text_dataset, batch_size=32, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 512
hidden_size = 512
vocab_size = len(word2index1)

num_layers = 1
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)
model.to(device)
# print(model)

num_epochs = 100

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for images, captions in train_dataloader:
        images = images.to(device)
        captions = captions.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, captions)
        targets = captions[:, 1:]
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'image_captioning_model.pth')









