# pytorch basic classification 
# Fashion mnist data set

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

label_tags = {
    0: 'T-Shirt', 
    1: 'Trouser', 
    2: 'Pullover', 
    3: 'Dress', 
    4: 'Coat', 
    5: 'Sandal', 
    6: 'Shirt',
    7: 'Sneaker', 
    8: 'Bag', 
    9: 'Ankle Boot' }
    
test_batch_size=1000
columns = 6
rows = 6

# CPU and MAIN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5) #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
        self.conv2 = nn.Conv2d(10, 30, 5)
        self.fc1 = nn.Linear(30 * 53 * 53, 1000)
        self.fc2 = nn.Linear(1000, 101)
        self.fc3 = nn.Linear(100,10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 30 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def inference(model, testset, device):
    fig = plt.figure(figsize=(10,10))
    plt.title(device, pad=50)
    for i in range(1, columns*rows+1):
        data_idx = np.random.randint(len(testset))
        input_img = testset[data_idx][0].unsqueeze(dim=0).to(device) 
        output = model(input_img)
        _, argmax = torch.max(output, 1)
        #pred = label_tags[argmax.item()]
        #label = label_tags[testset[data_idx][1]]
        
        fig.add_subplot(rows, columns, i)
        if (argmax.item() == testset[data_idx][1]):
            plt.title(str(argmax.item()) + ', right !!')
            cmap = 'Blues'
        else:
            plt.title('Not ' + str(argmax.item()) + ' but ' +  str(testset[data_idx][1]))
            cmap = 'Reds'
        plot_img = testset[data_idx][0][0,:,:]
        plt.imshow(plot_img, cmap=cmap)
        plt.axis('off')
    #plt.show() 




def main():
    epochs = 3
    learning_rate = 0.001
    batch_size = 32
    test_batch_size=16
    log_interval =100
    pth_path = "/home/yoon/Yoon/pytorch/research/only_gpu/test.pth"


    #print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    use_cuda = torch.cuda.is_available()
    print("use_cude : ", use_cuda)

    device = torch.device("cuda" if use_cuda else "cpu")

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
    testset = torchvision.datasets.Caltech101('./data',
        download=True,
        transform=transform)

    # model
    model = Net().to(device)

    # Freeze model weights
    for param in model.parameters():  # 전체 layer train해도 파라미터 안바뀌게 프리징
        param.requires_grad = False

    model.load_state_dict(torch.load(pth_path), strict=False) 
    model.eval()
    a = time.time()
    inference(model, testset, device)
    b = time.time()

    print("time : ", b - a)

if __name__ == '__main__':
    main()
