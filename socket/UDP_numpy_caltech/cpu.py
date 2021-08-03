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
RCV_HOST = '127.0.0.1'
RCV_PORT = 9999

SEND_HOST = '127.0.0.1'
SEND_PORT = 9998

MAX_PACKET_SIZE = 65503
BYTE_SIZE = 33
UDP_PAYLOAD_SIZE = MAX_PACKET_SIZE + BYTE_SIZE

send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#send_socket.bind((SEND_HOST, SEND_PORT))

receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_socket.bind((RCV_HOST, RCV_PORT))


def sender(x):
    snd = x.to("cpu").numpy().tobytes()
    bound = 0
    size = sys.getsizeof(snd)
    send_socket.sendto(str(size).encode(), (SEND_HOST, SEND_PORT))
    #print("사이즈 전송완료 : ", size)
    _ = receive_socket.recvfrom(1)
    #print("블록킹 팩터 통과")
    while(True):
        end = bound + MAX_PACKET_SIZE
        if (size < end): 
            send_socket.sendto(snd[bound:size], (SEND_HOST, SEND_PORT))
            break
        else: 
            send_socket.sendto(snd[bound:end], (SEND_HOST, SEND_PORT))
        bound = end 
        if not (size > MAX_PACKET_SIZE) : break

def receiver(tensor_shape):
    rcv = []
    rcv_size = 0

    size = receive_socket.recvfrom(1024)
    size = int(size[0].decode())
    print("사이즈 수신완료 : ", size )
    send_socket.sendto(b'', (SEND_HOST, SEND_PORT))
    #print("블록킹 팩터 전송")

    while(rcv_size < size-33) :
        data = receive_socket.recvfrom(65536)
        rcv.append(data[0])
        rcv_size += (sys.getsizeof(data[0])-33)
        print("rcv_size", rcv_size)
    rcv = b''.join(rcv)
    rcv = np.frombuffer(rcv, dtype=np.float32)
    rcv = np.reshape(rcv, tensor_shape)
    rcv = torch.from_numpy(rcv).to(device)
    return rcv

# CPU 
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

    def forward(self, x):
        x = receiver((1,3,224,224))
        x = self.conv1(x)
        sender(x)
        x = receiver((1,32,110,110))
        x = self.conv2(x)
        sender(x)
        x = receiver((1,64,53,53))
        x = self.conv3(x)
        sender(x)
        x = receiver((1,128,24,24))
        x = self.conv4(x)
        sender(x)

        return x


def inference(model, testset, device):
    for _ in range(1, columns*rows+1):
        input_img = testset[0][0].unsqueeze(dim=0).to(device) 
        _ = model(input_img)


def main():
    cpu_pth_path = "../../pth/caltech_cpu_3_5.pth"

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
    testset = torchvision.datasets.Caltech101('../../data',
        download=True,
        transform=transform)
    #testset = torchvision.datasets.ImageFolder('../../../data/caltech101', transform)

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
