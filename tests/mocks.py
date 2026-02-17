import contextlib


class MockWork:
    """Simulates a distributed async work handle."""

    def wait(self):
        pass


class MockEvent:
    """Simulates torch.cuda.Event."""

    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, end_event):
        return 0.0


class MockStream:
    """Simulates torch.cuda.Stream with context manager support."""

    def __init__(self, device=None, priority=0):
        self.device = device

    def wait_stream(self, stream):
        pass

    def record_event(self, event=None):
        return MockEvent()

    def synchronize(self):
        pass

    def wait_event(self, event):
        pass

    # --- Context Manager Protocol ---
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @contextlib.contextmanager
    def _use_stream(self):
        yield


# --- NVTX Mocking (For Profiler ranges) ---
class MockNVTX:
    @staticmethod
    def range_push(msg):
        pass

    @staticmethod
    def range_pop():
        pass


# --- Distributed Mocking ---
class MockDist:
    """
    Mocks torch.distributed functions to run NCCL code on CPU.
    """

    @staticmethod
    def all_to_all_single(output, input, group=None, async_op=False):
        # Identity communication: tokens stay on the same 'rank'
        if output.shape == input.shape:
            output.copy_(input)
        return MockWork() if async_op else None

    @staticmethod
    def get_world_size(group=None):
        return 2

    @staticmethod
    def get_rank(group=None):
        return 0

    def barrier(self, group=None):
        pass

    class group:
        WORLD = "MOCK_WORLD"
