import torch
import torch.multiprocessing as mp


def f(l):
    l.acquire()
    try:
        for num in range(100000):
            print('hello world', num)
    finally:
        l.release()

if __name__ == '__main__':
    lock = mp.Lock()

    f(lock)

