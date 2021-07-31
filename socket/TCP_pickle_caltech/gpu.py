from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import platform
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.font_manager._rebuild()
import numpy as np

import socket
import time
import sys
import _pickle

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
    a = time.time()
    snd = _pickle.dumps(x)
    b = time.time()
    print("pickle 시간 : ", b-a)
    bound = 0
    size = sys.getsizeof(snd)
    #print(">>>>>>>>>>", size)
    connection_socket.send(str(size).encode())
    a = connection_socket.recv(1)  # blocking factor
    #print(a)
    while(True):
        #print("sending index : ", bound)
        end = bound + MAX_PACKET_SIZE
        if (size < end): 
            connection_socket.send(snd[bound:size])
            break
        else: 
            connection_socket.send(snd[bound:end])
        bound = end 
        if not (size > MAX_PACKET_SIZE) : break

def receiver():
    rcv = []
    rcv_size = 0

    size = connection_socket.recv(8)
    #print(">>>>>>", size)
    size = int(size.decode())
    connection_socket.send(b'z')
    
    #print("total size : ", size)
    while(rcv_size < size-BYTE_SIZE) :
        #print("receive size : ", rcv_size)
        data = connection_socket.recv(UDP_PAYLOAD_SIZE)
        rcv.append(data)
        rcv_size += (sys.getsizeof(data)-BYTE_SIZE)

    rcv = b''.join(rcv)
    a = time.time()
    rcv = _pickle.loads(rcv).to(device)
    b = time.time()
    print("unpickle 시간 : ", b-a)
    return rcv

# CUDA 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5) #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
        self.conv2 = nn.Conv2d(20, 36, 5)
        self.fc1 = nn.Linear(60 * 53 * 53, 1000)
        self.fc2 = nn.Linear(1000, 101)

    def forward(self, x):
        # a = time.time()
        # sender(x)
        # b = time.time()
        # print("TCP Send duration : ", b-a)
        # a = time.time()
        # x = self.conv1(x)
        # b = time.time()
        # print("conv1 duration : ", b-a)
        # a = time.time()
        # y = receiver()
        # b = time.time()
        # print("TCP Receive duration : ", b-a)

        sender(x)
        x = self.conv1(x)
        y = receiver()


        x = torch.cat((x,y), 1)
        x = F.relu(x)
        x = self.pool(x)
        sender(x)
        x = self.conv2(x)
        y = receiver()
        x = torch.cat((x,y),1)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 60 * 53 * 53)
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
    gpu_pth_path = "../../pth/caltech_gpu_2_3.pth"

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
    testset = torchvision.datasets.Caltech101('../../data',
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
