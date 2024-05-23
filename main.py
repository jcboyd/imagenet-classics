import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from PIL import Image


def write_flush(*text_args, stream=sys.stdout):
    stream.write(', '.join(map(str, text_args)) + '\n')
    stream.flush()
    return

parser = argparse.ArgumentParser(description='ImageNet training script')
parser.add_argument('img_dir', type=str,
                    help='Image root directory.')
parser.add_argument('label_dir', type=str,
                    help='Label root directory.')
parser.add_argument('model', type=str,
                    help='Name of model architecture.')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to pretrained model.')
parser.add_argument('--img_size', type=int, default=224,
                    help='Size to rescale input images.')
parser.add_argument('--nb_epochs', type=int, default=1,
                    help='Number of epochs.')
parser.add_argument('--nb_batch', type=int, default=256,
                    help='Training batch size.')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Training learning rate.')
parser.add_argument('--decay', type=float, default=1e-5,
                    help='Weight decay strength.')
args = parser.parse_args()
write_flush(str(args))


def top_k_accuracy(outputs, y_val, k=1):

    """ Calculates top k accuracy for prediction

    Note top-k for k > 1 is only meaningful when modelling
    many similar classes.
    """

    top_k = torch.argsort(outputs, descending=True)[:, :k]
    in_top_k = torch.sum(top_k == y_val[:, None], axis=1)
    top_k_acc = torch.mean(in_top_k.double()).item()

    return top_k_acc


class ImageNetDataset(Dataset):

    def __init__(self, img_dir, label_dir, mode, transform=None, truncate=None):

        self.img_dir = os.path.join(img_dir, mode)

        label_file = f'{mode}_perm.txt'

        self.df_labels = pd.read_csv(os.path.join(label_dir, label_file),
                                     header=None,
                                     sep='\s+',
                                     names=['file', 'label'])

        if truncate is not None:  # take fraction of data for validation
            self.df_labels = self.df_labels[:truncate]

        self.transform = transform

    def __len__(self):
        return len(self.df_labels)

    def __getitem__(self, idx):

        row = self.df_labels.iloc[idx]
        index = row.file.index('_')
        n_dir, file_name = row.file[:index], row.file[index+1:]

        file_path = os.path.join(self.img_dir, n_dir, file_name)
        image = Image.open(file_path).convert('RGB')

        label = row.label

        if self.transform:
            image = self.transform(image)

        return image, label


class LogisticRegression(nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()

        self.input_shape = input_shape
        self.fc = nn.Linear(56 * 56 * 3, 1000)

    def forward(self, x):

        x = x.view(-1, 56 * 56 * 3)
        return self.fc(x)


class AlexNet_BN(nn.Module):

    def __init__(self):

        super(AlexNet_BN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), stride=4, padding=5),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, (5, 5), padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2, padding=1),
            nn.Conv2d(256, 384, (3, 3), padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2, padding=1),
            nn.Conv2d(384, 384, (3, 3), padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2, padding=1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(-1, 7 * 7 * 256)
        x = self.classifier(x)

        return x

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
write_flush(str(device))

T_tr = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])

T_te = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])

train_ds = ImageNetDataset(args.img_dir, args.label_dir, 'train', transform=T_tr)
train_loader = DataLoader(train_ds, batch_size=args.nb_batch, shuffle=True, drop_last=True, num_workers=10)

val_ds = ImageNetDataset(args.img_dir, './', 'val', transform=T_te, truncate=4096)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, drop_last=True, num_workers=10)

test_ds = ImageNetDataset(args.img_dir, './', 'val', transform=T_te)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=True, num_workers=10)

df_class_names = pd.read_csv(os.path.join(args.label_dir, 'labels.txt'), header=None)

model = globals()[args.model]().to(device)

for pp in model.parameters():
    nn.init.normal_(pp, std=1e-2)

if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))
    model.train()

opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

def learning_rate_schedule(epoch):

    if   epoch < 8 : return 1
    elif epoch < 11: return 0.1
    else:            return 0.01

scheduler = LambdaLR(opt, lr_lambda=learning_rate_schedule)

criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')

for epoch in range(args.nb_epochs):
    for batch_num, data in enumerate(train_loader):

        x_batch = data[0].to(device)
        y_batch = data[1].to(device)

        opt.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        opt.step()

        if batch_num % 100 == 0:

            model.eval()

            with torch.no_grad():

                val_data = [(model(data.to(device)), target) for data, target in val_loader]
                val_outputs = torch.cat([output for output, _ in val_data])
                val_targets = torch.cat([target for _, target in val_data]).to(device)

                top_1_acc = top_k_accuracy(val_outputs, val_targets, k=1)
                top_5_acc = top_k_accuracy(val_outputs, val_targets, k=5)

                val_loss = criterion(val_outputs, val_targets).item()  # N.B. item() important

                write_flush('[%04d] [%04d] tr_loss: %.4f val_loss: %.4f top_1: %.4f top_5: %.4f' % (
                    epoch, batch_num, loss, val_loss, top_1_acc, top_5_acc))

                if val_loss < best_val_loss:
                    torch.save(model.state_dict(), '%s.torch' % args.model)
                    best_val_loss = val_loss

            model.train()

    scheduler.step()

model = globals()[args.model]().to(device)
model.load_state_dict(torch.load('%s.torch' % args.model))
model.eval()

with torch.no_grad():

    test_data = [(model(data.to(device)).detach(), target) for data, target in test_loader]
    test_outputs = torch.cat([output for output, _ in test_data])
    test_targets = torch.cat([target for _, target in test_data]).to(device)

    top_1_acc = top_k_accuracy(test_outputs, test_targets, k=1)
    top_5_acc = top_k_accuracy(test_outputs, test_targets, k=5)

    write_flush('top_1: %.4f top_5: %.4f' % (top_1_acc, top_5_acc))
