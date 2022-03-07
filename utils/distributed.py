
from contextlib import closing
import socket

def find_free_port() -> int:
    """
    Find a free port for dist url
    :return:
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    return port

def scale_learning_rate(lr: float, world_size: int, batch_size: int, base_batch_size: int = 64) -> float:
    new_lr = lr * world_size * batch_size / base_batch_size
    print(f'adjust lr according to the number of GPU and batch size：{lr} -> {new_lr}')
    return new_lr
