# -*-coding:utf-8-*-
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])


class PreprocessDataset(Dataset):
    def __init__(self, imgPath, transform, ex=10):
        self.transforms = transform

        for _, _, files in os.walk(imgPath):

            self.imgs = [imgPath + '/' + file for file in files] * ex

        np.random.shuffle(self.imgs)

    def __len__(self):
        # print(self.imgs)
        return len(self.imgs)

    def __getitem__(self, index):

        tempImg = self.imgs[index]
        tempImg = Image.open(tempImg)

        sourceImg = self.transforms(tempImg)
        cropImg = torch.nn.MaxPool2d(4)(sourceImg)
        return cropImg, sourceImg


class ResBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)
        self.conv3 = nn.Conv2d(outChannels, outChannels, kernel_size=1, bias=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        resudial = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += resudial
        out = self.relu(out)
        return out


class SRResNet(nn.Module):
    """SRResNet模型(4x)"""

    def __init__(self):
        """初始化模型配置"""
        super(SRResNet, self).__init__()

        # 卷积模块1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, padding_mode='reflect', stride=1)
        self.relu = nn.PReLU()
        # 残差模块
        self.resBlock = self._makeLayer_(ResBlock, 64, 64, 16)
        # 卷积模块2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.PReLU()

        # 子像素卷积
        self.convPos1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=2, padding_mode='reflect')
        self.pixelShuffler1 = nn.PixelShuffle(2)
        self.reluPos1 = nn.PReLU()

        self.convPos2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.pixelShuffler2 = nn.PixelShuffle(2)
        self.reluPos2 = nn.PReLU()

        self.finConv = nn.Conv2d(64, 3, kernel_size=9, stride=1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []

        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        x = self.conv1(x)
        x = self.relu(x)
        residual = x

        out = self.resBlock(x)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.bn2(out)
        out += residual

        out = self.convPos1(out)
        # print(out.shape)
        out = self.pixelShuffler1(out)
        out = self.reluPos1(out)

        out = self.convPos2(out)
        # print(out.shape)
        out = self.pixelShuffler2(out)
        out = self.reluPos2(out)
        out = self.finConv(out)
        # print(out.shape)
        return out


def imshow(path, outputPath):
    preTransform = transforms.Compose([transforms.ToTensor()])
    img = Image.open(path)
    img = preTransform(img).unsqueeze(0)
    print(img.shape)
    net.cpu()
    source = net(img)[0, :, :, :]
    source = source.cpu().detach().numpy()
    print(source.shape)
    source = source.transpose((1, 2, 0))
    print(source.shape)
    source = np.clip(source, 0, 1)
    print(source.shape)
    img = Image.fromarray(np.uint8(source*255))
    img.save(outputPath)


def save_checkpoints_state(epoch, model, optimizer, model_param_save, run_name):
    checkpoint ={
        'epoch': epoch,
        "model_state_dict": model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if not os.path.isdir(model_param_save):
        os.mkdir(model_param_save)

    torch.save(checkpoint, os.path.join(model_param_save, run_name + '.pth'))


def get_checkpoints_state(dir, model, optimizer, run_name):

    print("Resume from checkpoint...")
    checkpoint = torch.load(os.path.join(dir, run_name+'.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('successfully recover from the last state')

    return model, epoch, optimizer


if __name__ == '__main__':

    path = 'URBAN100'

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    net = SRResNet()
    net.to(device)

    BATCH = 4
    processDataset = PreprocessDataset(imgPath=path, transform=transform)
    trainData = DataLoader(processDataset, batch_size=BATCH)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    lossF = nn.MSELoss().to(device)


    EPOCH = 30
    HISTORY = []
    for epoch in range(EPOCH):
        net.train()
        running_loss = 0

        for i, (cropImg, sourceImg) in tqdm(enumerate(trainData)):
            cropImg, sourceImg = cropImg.to(device), sourceImg.to(device)

            optimizer.zero_grad()

            outputs = net(cropImg)

            loss = lossF(outputs, sourceImg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average = running_loss / (i+1)
        HISTORY += [average]
        print('[INFO] %d loss: %.3f'% (epoch, average))

        running_loss = 0.0

    save_checkpoints_state(EPOCH, net, optimizer, 'checkpoint', '001')
    print('ok')
    # torch.save('dict.pickle')
    plt.plot(HISTORY, label='loss')
    plt.legend(loc='best')

    imshow('/home/hp/LZX/Image_Super/URBAN100/img001.png', '/home/hp/LZX/Image_Super/Output/img001.png')





