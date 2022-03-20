
from contextlib import closing
import socket
import io
import pickle
import torch
import torch.distributed as dist

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

def _object_to_tensor(obj):
    f = io.BytesIO()
    pickle.Pickler(f).dump(obj)
    byte_storage = torch.ByteStorage.from_buffer(f.getvalue()).tolist()
    byte_tensor = torch.tensor(byte_storage, dtype=torch.uint8)
    local_size = torch.tensor([byte_tensor.numel()], dtype=torch.long)
    return byte_tensor, local_size

def _tensor_to_object(tensor, tensor_size):
    buf = tensor.numpy().tobytes()[:tensor_size]
    return pickle.Unpickler(io.BytesIO(buf)).load()

def all_gather_object(object_list, obj, world_size):
    input_tensor, local_size = _object_to_tensor(obj)
    current_device = torch.device('cuda', torch.cuda.current_device())
    input_tensor = input_tensor.to(current_device)
    local_size = local_size.to(current_device)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    object_sizes_tensor = torch.zeros(world_size, dtype=torch.long, device=current_device)
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(world_size)
    ]
    # Allgather tensor sizes
    dist.all_gather(object_size_list, local_size)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * world_size, dtype=torch.uint8, device=current_device
    )
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(world_size)
    ]
    dist.all_gather(output_tensors, input_tensor)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8).cpu()  # type:ignore[call-overload]
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)
