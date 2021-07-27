import numpy as np
import socket
import sys

# Socket
HOST = '127.0.0.1'
PORT = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.bind((HOST, PORT))
client_socket.listen()
receive_socket, addr = client_socket.accept()
print('Connected by', addr)

send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
send_socket.connect((HOST, PORT))



MAX_PACKET_SIZE = 65503
BYTE_SIZE = 33
UDP_PAYLOAD_SIZE = MAX_PACKET_SIZE + BYTE_SIZE

def numpy_sender(x):
    snd = x.to("cpu").numpy().tobytes()
    print(snd)
    print(sys.getsizeof(snd))
    send_socket.send(snd)


def numpy_receiver():
    pass

