from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import platform
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.font_manager._rebuild()
import numpy as np

import socket
import time
import sys

from matplotlib import font_manager, rc
font_path = "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


device = "cuda"
columns = 6
rows = 6


# Socket
HOST = '127.0.0.1'
PORT = 9999

MAX_PACKET_SIZE = 65503
BYTE_SIZE = 33
UDP_PAYLOAD_SIZE = MAX_PACKET_SIZE + BYTE_SIZE

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()
connection_socket, addr = server_socket.accept()
print('Connected by', addr)

def sender(x):
    snd = x.to("cpu").numpy().tobytes()
    size = sys.getsizeof(snd)
    connection_socket.send(str(size).encode())
    _ = connection_socket.recv(1)  # blocking factor
    connection_socket.send(snd)

def receiver(tensor_shape):
    rcv = []
    rcv_size = 0

    size = connection_socket.recv(8)
    size = int(size.decode())-BYTE_SIZE
    connection_socket.send(b'z')

    while(rcv_size < size) :
        data = connection_socket.recv(min(size - rcv_size ,MAX_PACKET_SIZE))
        rcv.append(data)
        rcv_size += (sys.getsizeof(data)-BYTE_SIZE)

    rcv = b''.join(rcv)
    rcv = np.frombuffer(rcv, dtype=np.float32)
    rcv = np.reshape(rcv, tensor_shape)
    rcv = torch.from_numpy(rcv).to(device)

    return rcv

# CUDA 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5) #in, out, filtersize
        self.conv2 = nn.Conv2d(32, 40, 5) 
        self.conv3 = nn.Conv2d(64, 80, 5)
        self.conv4 = nn.Conv2d(128, 160, 5)
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling

        self.fc1 = nn.Linear(256 * 10 * 10, 1000)
        self.fc2 = nn.Linear(1000, 101)

    def forward(self, x):
        a = time.time()
        sender(x)
        b = time.time()
        print("TCP Send1 duration : ", b-a)

        a = time.time()
        x = self.conv1(x)
        b = time.time()
        print("conv1 duration : ", b-a)

        y = receiver((1,12,220,220))
        x = torch.cat((x,y), 1)
        x = F.relu(x)
        x = self.pool(x)

        a = time.time()
        sender(x)
        b = time.time()
        print("TCP Send2 duration : ", b-a)

        a = time.time()
        x = self.conv2(x)
        b = time.time()
        print("conv2 duration : ", b-a)

        y = receiver((1,24,106,106))
        x = torch.cat((x,y),1)
        x = F.relu(x)
        x = self.pool(x)

        a = time.time()
        sender(x)
        b = time.time()
        print("TCP Send3 duration : ", b-a)
        
        a = time.time()
        x = self.conv3(x)
        b = time.time()
        print("conv3 duration : ", b-a)

        y = receiver((1,48,49,49))
        x = torch.cat((x,y), 1)
        x = F.relu(x)
        x = self.pool(x)

        a = time.time()
        sender(x)
        b = time.time()
        print("TCP Send4 duration : ", b-a)

        a = time.time()
        x = self.conv4(x)
        b = time.time()
        print("conv4 duration : ", b-a)

        y = receiver((1,96,20,20))
        x = torch.cat((x,y), 1)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1,  256 * 10 * 10)
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
    gpu_pth_path = "../../../pth/caltech_gpu_3_5.pth"

    use_cuda = torch.cuda.is_available()
    print("use_cude : ", use_cuda)

    #device = torch.device("cuda" if use_cuda else "cpu")

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
