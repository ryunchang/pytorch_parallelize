# pytorch basic classification 
# Fashion mnist data set

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
from torch.multiprocessing import Process, Queue
import gc
gc.collect()
torch.cuda.empty_cache()

# CPU and MAIN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5) #in, out, filtersize
        self.conv2 = nn.Conv2d(32, 24, 5)
        self.conv3 = nn.Conv2d(64, 48, 5)
        self.conv4 = nn.Conv2d(128, 96, 5)
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling

        self.fc1 = nn.Linear(256 * 10 * 10, 1000)
        self.fc2 = nn.Linear(1000, 101)

        self.trash1 = torch.zeros((16, 20, 220, 220,)).to('cuda')
        self.trash2 = torch.zeros((16, 40, 106, 106,)).to('cuda')
        self.trash3 = torch.zeros((16, 80, 49, 49,)).to('cuda')
        self.trash4 = torch.zeros((16, 160, 20, 20,)).to('cuda')

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat((x,self.trash1), 1)
        x = F.relu(x)   
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.cat((x,self.trash2), 1)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = torch.cat((x,self.trash3), 1)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = torch.cat((x,self.trash4), 1)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 256 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CUDA 
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5) #in, out, filtersize
        self.conv2 = nn.Conv2d(32, 40, 5) 
        self.conv3 = nn.Conv2d(64, 80, 5)
        self.conv4 = nn.Conv2d(128, 160, 5)
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling

        self.fc1 = nn.Linear(256 * 10 * 10, 1000)
        self.fc2 = nn.Linear(1000, 101)

        self.trash1 = torch.zeros((16, 12, 220, 220,)).to("cuda")
        self.trash2 = torch.zeros((16, 24, 106, 106, )).to("cuda")
        self.trash3 = torch.zeros((16, 48, 49, 49,)).to("cuda")
        self.trash4 = torch.zeros((16, 96, 20, 20, )).to("cuda")

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat((x,self.trash1), 1)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.cat((x,self.trash2), 1)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = torch.cat((x,self.trash3), 1)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = torch.cat((x,self.trash4), 1)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 256 * 10 * 10)   
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss =0.0
    criterion = nn.CrossEntropyLoss() #defalut is mean of mini-batchsamples
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #print(target)
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

def my_run(args):
    start_time = time.time()
    for epoch in range(1, args[6] + 1):
        print(args[2], "??????", epoch, "??? ?????? ??????")
        train(args[0], args[1], args[2], args[3], args[5], epoch)
        test(args[1], args[2], args[4])
    stop_time = time.time()
    print("duration : ", stop_time - start_time)


def main():
    epochs = 9
    learning_rate = 0.001
    batch_size = 16
    test_batch_size = 16
    log_interval = 100

    #print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    use_cuda = torch.cuda.is_available()
    print("use_cude : ", use_cuda)

    #device = torch.device("cuda" if use_cuda else "cpu")
    device1 = "cuda"
    device2 = "cuda"

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
    
    print(trainset)
    print(testset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=nThreads)


    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                            shuffle=False, num_workers=nThreads)

    # model
    model1 = Net().to(device1)
    model1.share_memory()
    model2 = Net2().to(device2)
    model2.share_memory()
    

    # Freeze model weights
    # for param in model1.parameters():  # ?????? layer train?????? ???????????? ???????????? ?????????
    #     param.requires_grad = False
    # for param in model2.parameters():  # ?????? layer train?????? ???????????? ???????????? ?????????
    #     param.requires_grad = False
        
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer1 = optim.Adam(model1.parameters(),lr=learning_rate)
    optimizer2 = optim.Adam(model2.parameters(),lr=learning_rate)

    device_args1 = (log_interval, model1, device1, train_loader, test_loader, optimizer1, epochs)
    device_args2 = (log_interval, model2, device2, train_loader, test_loader, optimizer2, epochs)

    proc1 = Process(target=my_run, args=(device_args1,))
    proc2 = Process(target=my_run, args=(device_args2,))
    
    num_processes = (proc1, ) 
    processes = []
    
    for procs in num_processes:
        procs.start()
        processes.append(procs)
    
    for proc in processes:
        proc.join()

    # Save model
    torch.save(model1.state_dict(), "../../pth/caltech_cpu_3_5.pth")

    # Save model
    #torch.save(model2.state_dict(), "../../pth/caltech_gpu_3_5.pth")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')# good solution !!!
    main()
