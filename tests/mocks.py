import contextlib


class MockStream:
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

    @contextlib.contextmanager
    def _use_stream(self):
        yield


class MockEvent:
    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, end_event):
        return 0.0


class MockStreamManager:
    def __init__(self, device):
        self.device = device
        self.compute_stream = MockStream(device)
        self.comm_stream = MockStream(device)

    def wait_comm(self):
        pass

    def wait_compute(self):
        pass
