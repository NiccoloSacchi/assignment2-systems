import torch


def print_gpu_memory():
    # Convert bytes to megabytes
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**2)

    print("Memory:")
    print(f"\tAllocated: {allocated:.1f} MB")
    print(f"\tReserved: {reserved:.1f} MB")
    print(f"\tTotal: {total:.4f} MB")
