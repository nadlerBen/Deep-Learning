import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import shutil
# import make_graph
batch_size = 64
epochs = 300
seed = 1
opt = 'sgd'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)


    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(Bottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out

def main():

    use_gpu = torch.cuda.is_available()
    save_path = 'work/densenet.base'

    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed(seed)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    channel_means = [0.4395, 0.4298, 0.3977]
    channel_stds = [0.2761, 0.2714, 0.2799]

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(channel_means, channel_stds)
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(channel_means, channel_stds)
    ])

    trainLoader = DataLoader(
        dset.CIFAR10(root='./data/', train=True, download=True,transform=trainTransform),
        batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(
        dset.CIFAR10(root='./data/', train=False, download=True,transform=testTransform),
        batch_size=batch_size, shuffle=False)

    net = DenseNet(growthRate=8, depth=30, reduction=0.5, nClasses=10)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if use_gpu:
        net = net.cuda()

    if opt == 'sgd':
        optimizer = optim.SGD(net.parameters(),lr=1e-2,momentum=0.9, weight_decay=1e-4)
    elif opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        adjust_opt(opt, optimizer, epoch)
        train(use_gpu, epoch, net, trainLoader, optimizer)
        _test(use_gpu, net, testLoader)
        torch.save(net, os.path.join(save_path, 'latest.pkl'))


def train(use_gpu, epoch, net, trainLoader, optimizer):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        if (batch_idx + 1) % 10 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Accuracy: %.4f'
                  % (epoch , epochs, batch_idx + 1, nTrain // batch_size, loss.item(),100-err))

def _test(use_gpu, net, testLoader):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(test_loss, 100-err))


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-2
        elif epoch == 150: lr = 1e-3
        elif epoch == 225: lr = 1e-4
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()