import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

transform = transforms.Compose(
         [transforms.ToTensor(),
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
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)


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


model = torch.load('bestModel.pkl')

correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28 * 28)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the model on the 10000 test images: %.4f %%' % (float(correct) / total))
