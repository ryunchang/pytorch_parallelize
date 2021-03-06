# pytorch basic classification 
# Fashion mnist data set

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import time
import socket
from torch.multiprocessing import Process, Queue
import _pickle
import sys

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

# Socket
RCV_HOST = '127.0.0.1'
RCV_PORT = 9999

SEND_HOST = '127.0.0.1'
SEND_PORT = 9998

send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#send_socket.bind((SEND_HOST, SEND_PORT))

receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_socket.bind((RCV_HOST, RCV_PORT))

MAX_PACKET_SIZE = 65503

def packet_sender(x):
    snd = _pickle.dumps(x)
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

def packet_receiver() :
    rcv = []
    rcv_size = 0

    size = receive_socket.recvfrom(1024)
    size = int(size[0].decode())
    #print("사이즈 수신완료 : ", size )
    send_socket.sendto(b'', (SEND_HOST, SEND_PORT))
    #print("블록킹 팩터 전송")

    while(rcv_size < size-33) :
        data = receive_socket.recvfrom(65536)
        rcv.append(data[0])
        rcv_size += (sys.getsizeof(data[0])-33)
        #print("rcv_size", rcv_size)
    rcv = b''.join(rcv)
    return rcv 

# CPU 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 8, 5)     #in, out, filtersize
        # self.pool = nn.MaxPool2d(2, 2)  #2x2 pooling
        # self.conv2 = nn.Conv2d(20, 24, 5)
        # self.fc1 = nn.Linear(60 * 53 * 53, 1000)
        # self.fc2 = nn.Linear(1000, 101)
        self.conv1 = nn.Conv2d(3, 2, 5)     #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2)  #2x2 pooling
        self.conv2 = nn.Conv2d(10, 6, 5)
        self.fc1 = nn.Linear(30 * 53 * 53, 1000)
        self.fc2 = nn.Linear(1000, 101)


    def forward(self, x):

        rcv = packet_receiver()

        print("1수신 완료")

        x = _pickle.loads(rcv).to('cpu')
        x = self.conv1(x)

        packet_sender(x)
        print("2전송 완료")
        

        rcv = packet_receiver()
        print("3수신 완료")

        x = _pickle.loads(rcv).to('cpu')
        x = self.conv2(x)

        packet_sender(x)
        print("4전송 완료")

        # rcv = packet_receiver()
        # print("5수신 완료")

        # x = pickle.loads(rcv).to('cpu')

        return x


def inference(model, testset, device):
    fig = plt.figure(figsize=(10,10))
    plt.title(device, pad=50)
    for i in range(1, columns*rows+1):
        print("-----------------------------")
        data_idx = np.random.randint(len(testset))
        input_img = testset[data_idx][0].unsqueeze(dim=0).to(device) 
        output = model(input_img)
    #     _, argmax = torch.max(output, 1)
    #     pred = label_tags[argmax.item()]
    #     label = label_tags[testset[data_idx][1]]

    #     fig.add_subplot(rows, columns, i)
    #     if pred == label:
    #         plt.title(pred + ', right !!')
    #         cmap = 'Blues'
    #     else:
    #         plt.title('Not ' + pred + ' but ' +  label)
    #         cmap = 'Reds'
    #     plot_img = testset[data_idx][0][0,:,:]
    #     plt.imshow(plot_img, cmap=cmap)
    #     plt.axis('off')
    #     print("-----------------------------")
    # plt.show()      # If you want to measure inferencing time, comment out this line


def main():

    #cpu_pth_path = "../../../pth/caltech_cpu_2080.pth"
    cpu_pth_path = "../../../pth/caltech_cpu.pth"
    gpu_pth_path = "../../../pth/caltech_gpu_2080.pth"

    #print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    use_cuda = torch.cuda.is_available()
    print("use_cude : ", use_cuda)

    #device = torch.device("cuda" if use_cuda else "cpu")
    device1 = "cpu"
    device2 = "cuda"

    # nThreads = 1 if use_cuda else 2 
    # if platform.system() == 'Windows':
    #     nThreads =0 #if you use windows

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
#    testset = torchvision.datasets.Caltech101('../../../data',
#        download=True,
#        transform=transform)
    testset = torchvision.datasets.ImageFolder('../../../data', transform)

    # test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
    #                                         shuffle=False, num_workers=nThreads)


    # model
    model = Net().to(device1)
    #model.share_memory()    # imshow example

    model.load_state_dict(torch.load(cpu_pth_path), strict=False) 
    model.eval()

    # Freeze model weights
    for param in model.parameters():  # 전체 layer train해도 파라미터 안바뀌게 프리징
        param.requires_grad = False

    a = time.time()
    inference(model, testset, device1)
    b = time.time()
    print("time : ", b - a)


if __name__ == '__main__':
    main()
