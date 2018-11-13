import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image

DATA_FOLDER = sys.argv[1]
BATCH_SIZE = 64

class NumbersDataset(Dataset):
    def __init__(self, data_folder):
        self.fnames = []
        for (root, dirs, files) in os.walk(data_folder):
            for filename in files:
                self.fnames.append(os.path.join(root, filename))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        filename = self.fnames[idx]
        img = Image.open(filename).convert('L')
        img_as_np = np.asarray(img)
        img_as_np = img_as_np.reshape(1,28,28)
        sample = {'img': img_as_np, 'label': int(filename.split('/')[-2].split('_')[-1])}
        return sample

train_folder = DATA_FOLDER + '/train'
valid_folder = DATA_FOLDER + '/valid'
train_dataset = NumbersDataset(train_folder)
valid_dataset = NumbersDataset(valid_folder)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=9999)

class NumberClassifier(nn.Module):
    def __init__(self):
        super(NumberClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NumberClassifier().to(device)
optimizer = optim.Adam(model.parameters())
# optimizer = optim.SGD(model.parameters(), 0.01, momentum=0.5)
print_train_interval = len(train_dataloader) // 5
for epoch in range(10):
    print('=====EPOCH {}====='.format(epoch+1))
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        imgs, labels = batch['img'].to(device), batch['label'].to(device)
        imgs = imgs.float() / 255
        output = model(imgs)
        loss = F.nll_loss(output, labels)
        if (batch_idx+1) % print_train_interval == 0:
            print('train_loss:', loss)
#             print('output:', output.argmax(dim=1))
#             print('labels:', labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_loss = 0
    val_acc = 0
    for batch in valid_dataloader:
        imgs, labels = batch['img'].to(device), batch['label'].to(device)
        imgs = imgs.float() / 255
        output = model(imgs)
        val_loss += F.nll_loss(output, labels, reduction='sum') / len(valid_dataset)
        val_acc += torch.sum(labels == output.argmax(dim=1)).float() / len(valid_dataset)
#     print('output:', output.argmax(dim=1))
#     print('labels:', labels)
    print()
    print('val_loss:', val_loss)
    print('val_acc:', val_acc)

torch.save(model.state_dict(), 'model.pth')
print('Saved model: model.pth')

