from typing import Any
from collections import OrderedDict
import torch.accelerator.memory as accel_mem
import torch


def _allocator_ready() -> bool:
    try:
        return torch._C._accelerator_isAllocatorInitialized()
    except RuntimeError:
        return False


def memory_stats(device_index: int | None = None) -> OrderedDict[str, Any]:
    if not _allocator_ready():
        return OrderedDict()
    return accel_mem.memory_stats(device_index)


def memory_allocated(device_index: int | None = None) -> int:
    if not _allocator_ready():
        return 0
    return accel_mem.memory_allocated(device_index)


def max_memory_allocated(device_index: int | None = None) -> int:
    if not _allocator_ready():
        return 0
    return accel_mem.max_memory_allocated(device_index)


def reset_peak_memory_stats(device_index: int | None = None) -> None:
    if not _allocator_ready():
        return
    accel_mem.reset_peak_memory_stats(device_index)


def reset_accumulated_memory_stats(device_index: int | None = None) -> None:
    if not _allocator_ready():
        return
    accel_mem.reset_accumulated_memory_stats(device_index)
