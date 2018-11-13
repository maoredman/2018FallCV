import os, csv, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

TEST_FOLDER = sys.argv[1]
OUT_NAME = sys.argv[2]
NUM_TEST = 10000
TEST_BATCH = 9999

class TestDataset(Dataset):
    def __init__(self, data_folder):
        self.fnames = ['{}/{}.png'.format(data_folder, str(i).zfill(4)) for i in range(NUM_TEST)]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        filename = self.fnames[idx]
        img = Image.open(filename).convert('L')
        img_as_np = np.asarray(img)
        img_as_np = img_as_np.reshape(1,28,28)
        sample = {'img': img_as_np}
        return sample

test_dataset = TestDataset(TEST_FOLDER)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH)

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
model.load_state_dict(torch.load('model.pth'))
model.eval()

preds = torch.LongTensor()
for batch in test_dataloader:
    imgs = batch['img'].to(device)
    imgs = imgs.float() / 255
    preds = torch.cat((preds, model(imgs).argmax(dim=1)))

with open(OUT_NAME, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])
    for i in range(NUM_TEST):
        writer.writerow([str(i).zfill(4), int(preds[i])])

