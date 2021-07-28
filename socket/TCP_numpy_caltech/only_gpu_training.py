from numpy.core.fromnumeric import shape
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import platform
import matplotlib.pyplot as plt
import numpy as np

import time


columns = 6
rows = 6

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5) #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
        self.conv2 = nn.Conv2d(20, 60, 5)
        self.fc1 = nn.Linear(60 * 53 * 53, 1000)
        self.fc2 = nn.Linear(1000, 101)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 60 * 53 * 53)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss =0.0
    criterion = nn.CrossEntropyLoss() #defalut is mean of mini-batchsamples
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(device, ' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss/log_interval))
            running_loss =0.0


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion =  nn.CrossEntropyLoss(reduction='sum') #add all samples in a mini-batch
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss +=  loss.item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    epochs = 8
    learning_rate = 0.001
    batch_size = 16
    test_batch_size = 16
    log_interval = 100

    use_cuda = torch.cuda.is_available()
    print("use_cude : ", use_cuda)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("device : ", device)

    nThreads = 1 if use_cuda else 2 
    if platform.system() == 'Windows':
        nThreads =0 #if you use windows
  
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])

    # datasets
    trainset = torchvision.datasets.Caltech101('../../data',
        download=True,
        transform=transform)
    testset = torchvision.datasets.Caltech101('../../data',
        download=True,
        transform=transform)
    
    # trainset = torchvision.datasets.ImageFolder('../../data/caltech101', transform)
    # testset = torchvision.datasets.ImageFolder('../../data/caltech101', transform) 
    

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=nThreads)


    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                            shuffle=False, num_workers=nThreads)

    # model
    model = Net().to(device)

    # Freeze model weights
    # for param in model1.parameters():  # 전체 layer train해도 파라미터 안바뀌게 프리징
    #     param.requires_grad = False
    # for param in model2.parameters():  # 전체 layer train해도 파라미터 안바뀌게 프리징
    #     param.requires_grad = False
        
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    start_time = time.time()
    for epoch in range(1,  epochs + 1):
        print(device, "에서", epoch, "회 에폭 실행")
        train(log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device,  test_loader)
    stop_time = time.time()
    print("duration : ", stop_time - start_time)

    # Save model
    torch.save(model.state_dict(), "../../pth/caltech_only_gpu.pth")


if __name__ == '__main__':
    main()
