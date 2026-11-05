# Adapted for WAPR-realsense; SPDX-License-Identifier: LGPL-2.1-only
"""One enqueue-memory buffer for TensorRT contexts that execute in sequence."""

from contextlib import contextmanager
import threading

import torch


class SharedExecutionMemory:
    """Keep engine weights/persistent state separate; share only enqueue memory.

    Register all engine requirements before the first inference to allocate once.
    The lock serializes host submissions. CUDA stream ordering protects reuse on
    one stream; an event synchronizes stream changes and buffer growth.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TensorRT execution memory requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.required_bytes = 0
        self.allocated_bytes = 0
        self._buffer = None
        self._event = None
        self._stream_handle = None
        self._lock = threading.RLock()

    def reserve(self, size):
        with self._lock:
            self.required_bytes = max(self.required_bytes, int(size))

    @contextmanager
    def use(self, context, device, stream):
        if torch.device(device) != self.device:
            raise ValueError("TensorRT shared memory and inputs must use the same CUDA device")
        with self._lock, torch.cuda.device(self.device):
            grow = self._buffer is None or self.allocated_bytes < self.required_bytes
            if self._event is not None and (grow or self._stream_handle != stream.cuda_stream):
                self._event.synchronize()
            if grow:
                # Completed submissions no longer need the old buffer. Release it
                # before allocating a larger one to avoid doubling the live size.
                self._buffer = None
                self.allocated_bytes = 0
                self._buffer = torch.empty(max(1, self.required_bytes) + 255,
                                           dtype=torch.uint8, device=self.device)
                self.allocated_bytes = max(1, self.required_bytes)
                print("WAPR_TRT_MEMORY", {"device": str(self.device),
                      "shared_execution_mib": round(self.allocated_bytes / 1024 ** 2, 2)}, flush=True)
            # TensorRT 8 requires 256-byte alignment. Retain the owning tensor,
            # including padding, for the entire asynchronous execution lifetime.
            pointer = int(self._buffer.data_ptr())
            context.device_memory = (pointer + 255) & ~255
            self._buffer.record_stream(stream)
            try:
                yield
            finally:
                if self._event is None:
                    self._event = torch.cuda.Event()
                self._event.record(stream)
                self._stream_handle = stream.cuda_stream
