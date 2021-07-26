import _pickle
import socket
import sys

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


if (__name__ == '__main__'):
    pass
