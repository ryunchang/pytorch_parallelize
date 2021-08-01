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
        self.conv1 = nn.Conv2d(3, 32, 5) #in, out, filtersize
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling

        self.fc1 = nn.Linear(256 * 10 * 10, 1000)
        self.fc2 = nn.Linear(1000, 101)

    def forward(self, x):
        a = time.time()
        x = self.conv1(x)
        b = time.time()
        print("conv1 duration : ", b-a)

        x = F.relu(x)
        x = self.pool(x)
        a = time.time()
        x = self.conv2(x)
        b = time.time()
        print("conv1 duration : ", b-a)

        x = F.relu(x)
        x = self.pool(x)
        a = time.time()
        x = self.conv3(x)
        b = time.time()
        print("conv1 duration : ", b-a)

        x = F.relu(x)
        x = self.pool(x)
        a = time.time()
        x = self.conv4(x)
        b = time.time()
        print("conv1 duration : ", b-a)

        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 256 * 10 * 10)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

def inference(model, testset, device):
    fig = plt.figure(figsize=(10,10))
    plt.title(device, pad=50)
    for i in range(1, columns*rows+1):
        print("-----------------------------")
        data_idx = np.random.randint(len(testset))
        input_img = testset[data_idx][0].unsqueeze(dim=0).to(device) 
        output = model(input_img)
        _, argmax = torch.max(output, 1)
        pred = str(argmax.item()) #label_tags[argmax.item()]
        label = str(testset[data_idx][1]) #label_tags[testset[data_idx][1]]
        fig.add_subplot(rows, columns, i)
        if pred == label:
            plt.title(pred + ', right !!')
            cmap = 'Blues'
        else:
            plt.title('Not ' + pred + ' but ' +  label)
            cmap = 'Reds'
        plot_img = testset[data_idx][0][0,:,:]
        plt.imshow(plot_img, cmap=cmap)
        plt.axis('off')
        print("-----------------------------")
    plt.show()      # If you want to measure inferencing time, comment out this line


def main():
    #gpu_pth_path = "../../../pth/caltech_gpu_2080.pth"
    gpu_pth_path = "../../../pth/caltech_only_3_5_gpu.pth"

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
    testset = torchvision.datasets.Caltech101('../../../data',
        download=True,
        transform=transform)
    #testset = torchvision.datasets.ImageFolder('../../data/caltech101', transform)

    # model
    model = Net().to(device)

    model.load_state_dict(torch.load(gpu_pth_path), strict=False) 
    model.eval()
    
    # Freeze model weights
    for param in model.parameters():  # 전체 layer train해도 파라미터 안바뀌게 프리징
        param.requires_grad = False

    a = time.time()
    inference(model, testset, device)
    b = time.time()
    print("time : ", b - a)

if __name__ == '__main__':
    main()
