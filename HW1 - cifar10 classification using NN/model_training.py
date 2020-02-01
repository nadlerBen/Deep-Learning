import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np

# Hyper Parameters
input_size = 784
hidden_size = 70
num_classes = 10
num_epochs = 15
batch_size = 100
learning_rate = 0.00667559446404323
reg = 1.8238438010732052e-05
momentum = 0.7946674647068385
momentum2 = 0.948441462819717

transform = transforms.Compose(
         [
             transforms.ToTensor(),
         transforms.Normalize((0.28604063,), (0.32045338,))])


# Fashion-MNIST Dataset (Images and Labels)
train_dataset = dsets.FashionMNIST(root='./data',
                            train=True,
                            transform=transform,
                            download=True)

test_dataset = dsets.FashionMNIST(root='./data',
                           train=False,
                           transform=transform)

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


train_loader = DeviceDataLoader(train_loader, device)
test_loader = DeviceDataLoader(test_loader, device)

# Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)


net = NeuralNet(input_size, num_classes)
net.to(device)
net.apply(init_weights)

# Loss and Optimizer
# Softmax is internally computed.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=reg, betas=(momentum, momentum2))

print(f'Number of parameters: {sum(param.numel() for param in net.parameters())}')
print(f'Num of trainable parameters : {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

(images, labels) = next(iter(train_loader))
for idx, image in enumerate(images):
    if np.random.uniform(0, 1) < 0.5:
        images[idx] = image.flip(1, 2)

# Training the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28 * 28)
        labels = labels

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28 * 28)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: %.4f %%' % (float(correct) / total))




