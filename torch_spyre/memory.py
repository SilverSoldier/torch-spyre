from typing import Any
import collections

__all__ = [
    "memory_allocated",
    "max_memory_allocated"
]


def memory_allocated(device: int | None = None) -> int:
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)

def max_memory_allocated(device: int | None = None) -> int:
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)

def memory_stats(device: int | None = None) -> dict[str, Any]:
    r"""Return a dictionary of Spyre memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all}.{current,peak,allocated,freed}"``:
      amount of allocated memory.

    Pool type:

    - ``all``: combined statistics across all memory pools.

     Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)

def memory_stats_as_nested_dict(device: int | None = None) -> dict[str, Any]:
    r"""Return the result of :func:`~torch.spyre.memory_stats` as a nested dictionary."""
    import torch_spyre._C as _C
    return _C.memory_stats(device)

