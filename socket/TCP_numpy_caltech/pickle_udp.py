import _pickle
import socket
import sys

# Socket
RCV_HOST = '127.0.0.1'
RCV_PORT = 9999

SEND_HOST = '127.0.0.1'
SEND_PORT = 9998

send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_socket.bind((RCV_HOST, RCV_PORT))

MAX_PACKET_SIZE = 65503
BYTE_SIZE = 33
UDP_PAYLOAD_SIZE = MAX_PACKET_SIZE + BYTE_SIZE

def packet_sender(x):
    snd = _pickle.dumps(x)
    bound = 0
    size = sys.getsizeof(snd)
    send_socket.sendto(str(size).encode(), (SEND_HOST, SEND_PORT))
    _ = receive_socket.recvfrom(1)  # blocking factor

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

    size = receive_socket.recvfrom(64)
    size = int(size[0].decode())
    send_socket.sendto(b'', (SEND_HOST, SEND_PORT))

    while(rcv_size < size-BYTE_SIZE) :
        data = receive_socket.recvfrom(UDP_PAYLOAD_SIZE)
        rcv.append(data[0])
        rcv_size += (sys.getsizeof(data[0])-BYTE_SIZE)

    rcv = b''.join(rcv)
    return rcv 


if (__name__ == '__main__'):
    pass
