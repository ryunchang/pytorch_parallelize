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

# CPU 
class Net(nn.Module):
    def __init__(self, shared_queue1, shared_queue2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5) #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
        self.conv2 = nn.Conv2d(100, 60, 5)
        self.fc1 = nn.Linear(300 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.fc3 = nn.Linear(100,10)
        self.q1 = shared_queue1     # CPU -> GPU
        self.q2 = shared_queue2     # GPU -> CPU

    def forward(self, x):
        x = self.q2.get(True, None).to("cpu")
        x = self.conv1(x)
        self.q1.put(x)
        x = self.q2.get(True, None).to("cpu")
        x = self.conv2(x)
        self.q1.put(x)
        x = self.q2.get(True, None).to("cpu")
        return x

# CUDA AND MAIN
class Net2(nn.Module):
    def __init__(self, shared_queue1, shared_queue2):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 80, 5) #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
        self.conv2 = nn.Conv2d(100, 240, 5)
        self.fc1 = nn.Linear(300 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.fc3 = nn.Linear(100,10)
        self.q1 = shared_queue1     # CPU -> GPU
        self.q2 = shared_queue2     # GPU -> CPU

    def forward(self, x):
        self.q2.put(x)
        x = self.conv1(x)
        y = self.q1.get(True, None).to("cuda")
        x = torch.cat((x,y), 1)
        x = self.pool(F.relu(x))
        self.q2.put(x)
        x = self.conv2(x)
        y = self.q1.get(True, None).to("cuda")
        x = torch.cat((x,y),1)
        x = self.pool(F.relu(x))
        x = x.view(-1, 300 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        self.q2.put(x)
        return x


def trash_inference(model, testset, device):
    for _ in range(1, columns*rows+1):
        input_img = testset[0][0].unsqueeze(dim=0).to(device) 
        _ = model(input_img)


def inference(model, testset, device):
    fig = plt.figure(figsize=(10,10))
    plt.title(device, pad=50)
    for i in range(1, columns*rows+1):
        print("-----------------------------")
        data_idx = np.random.randint(len(testset))
        input_img = testset[data_idx][0].unsqueeze(dim=0).to(device) 
        output = model(input_img)
        _, argmax = torch.max(output, 1)
        pred = label_tags[argmax.item()]
        label = label_tags[testset[data_idx][1]]

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


def my_run(model, testset, device, pth_path):
    model.load_state_dict(torch.load(pth_path), strict=False) 
    model.eval()
    if device == "cuda":
        a = time.time()
        inference(model, testset, device)
        b = time.time()
        print("time : ", b - a)
    else :
        trash_inference(model, testset, device)


def main():
    q1 = Queue()    # CPU -> GPU
    q2 = Queue()    # GPU -> CPU

    cpu_pth_path = "/home/yoon/Yoon/pytorch_research/pth/fashionMnist/cpu_20_80.pth"
    gpu_pth_path = "/home/yoon/Yoon/pytorch_research/pth/fashionMnist/gpu_20_80.pth"

    #print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    use_cuda = torch.cuda.is_available()
    print("use_cude : ", use_cuda)

    #device = torch.device("cuda" if use_cuda else "cpu")
    device1 = "cpu"
    device2 = "cuda"

    nThreads = 1 if use_cuda else 2 
    if platform.system() == 'Windows':
        nThreads =0 #if you use windows

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # datasets
    testset = torchvision.datasets.FashionMNIST('../data',
        download=True,
        train=False,
        transform=transform)


    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # model
    model1 = Net(q1, q2).to(device1)
    model1.share_memory()    # imshow example
    model2 = Net2(q1, q2).to(device2)
    model2.share_memory()

    # Freeze model weights
    for param in model1.parameters():  # 전체 layer train해도 파라미터 안바뀌게 프리징
        param.requires_grad = False
    for param in model2.parameters():  # 전체 layer train해도 파라미터 안바뀌게 프리징
        param.requires_grad = False

    proc1 = Process(target=my_run, args=(model1, testset, device1, cpu_pth_path))
    proc2 = Process(target=my_run, args=(model2, testset, device2, gpu_pth_path))

    num_processes = (proc2, proc1) 
    processes = []

    for procs in num_processes:
        procs.start()
        processes.append(procs)

    for proc in processes:
        proc.join()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')# good solution !!!
    main()
