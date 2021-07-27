import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import time
import sys
import socket

device = "cpu"
columns = 6
rows = 6

# Socket
HOST = '127.0.0.1'
PORT = 9999

MAX_PACKET_SIZE = 65503
BYTE_SIZE = 33
UDP_PAYLOAD_SIZE = MAX_PACKET_SIZE + BYTE_SIZE

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))


def sender(x):
    snd = x.to("cpu").numpy().tobytes()
    bound = 0
    size = sys.getsizeof(snd)
    print(">>>>>>>>>>", size)
    client_socket.send(str(size).encode())
    _ = client_socket.recv(1)  # blocking factor

    while(True):
        print("sending index : ", bound)
        end = bound + MAX_PACKET_SIZE
        if (size < end): 
            client_socket.send(snd[bound:size])
            break
        else: 
            client_socket.send(snd[bound:end])
        bound = end 
        if not (size > MAX_PACKET_SIZE) : break

def receiver(tensor_shape):
    rcv = []
    rcv_size = 0

    size = client_socket.recv(8)
    print(">>>>>>", size)
    size = int(size.decode())
    client_socket.send(b'z')
    
    print("total size : ", size)
    while(rcv_size < size-BYTE_SIZE) :
        print("receive size : ", rcv_size)
        data = client_socket.recv(UDP_PAYLOAD_SIZE)
        #print(data[0])
        rcv.append(data)
        rcv_size += (sys.getsizeof(data)-BYTE_SIZE)

    rcv = b''.join(rcv)
    rcv = np.frombuffer(rcv, dtype=np.float32)
    rcv = np.reshape(rcv, tensor_shape)
    rcv = torch.from_numpy(rcv).to(device)
    return rcv

# CPU 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)     #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2)  #2x2 pooling
        self.conv2 = nn.Conv2d(20, 24, 5)
        self.fc1 = nn.Linear(60 * 53 * 53, 1000)
        self.fc2 = nn.Linear(1000, 101)

    def forward(self, x):
        x = receiver((1,3,224,224))
        x = self.conv1(x)
        sender(x)
        x = receiver((1,20,110,110))
        x = self.conv2(x)
        sender(x)
        
        # rcv = packet_receiver()
        # x = pickle.loads(rcv).to(device)

        return x


def inference(model, testset, device):
    for _ in range(1, columns*rows+1):
        input_img = testset[0][0].unsqueeze(dim=0).to(device) 
        _ = model(input_img)


def main():
    cpu_pth_path = "../../pth/caltech_cpu_2080.pth"

    use_cuda = torch.cuda.is_available()
    print("use_cude : ", use_cuda)

    #device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) ])

    #datasets
    testset = torchvision.datasets.Caltech101('../../data/caltech101',
        download=True,
        transform=transform)
    #testset = torchvision.datasets.ImageFolder('../../data/caltech101', transform)

    # model
    model = Net().to(device)

    model.load_state_dict(torch.load(cpu_pth_path), strict=False) 
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
