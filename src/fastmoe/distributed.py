from abc import ABC, abstractmethod

import torch
import torch.distributed as dist


class Communicator(ABC):
    @abstractmethod
    def exchange_counts(self, local_counts: torch.Tensor) -> torch.Tensor:
        """
        Exchange the number of tokens each rank intends to send to others.
        Input: [World_Size] (Int)
        Output: [World_Size] (Int) - How many I will receive from each rank.
        """
        pass

    @abstractmethod
    def all_to_all(
        self, x: torch.Tensor, send_counts: list[int], recv_counts: list[int]
    ) -> torch.Tensor:
        """
        The heavy data movement.
        """
        pass


class PytorchCommunicator(Communicator):
    def __init__(self, group=None):
        self.group = group
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def exchange_counts(self, local_counts: torch.Tensor) -> torch.Tensor:
        """
        All-to-All for scalar counts.
        We use 'all_to_all_single' on a small tensor.
        """
        if self.world_size == 1:
            return local_counts

        # Buffer to receive counts from everyone
        recv_counts = torch.zeros_like(local_counts)
        dist.all_to_all_single(recv_counts, local_counts, group=self.group)
        return recv_counts

    def all_to_all(
        self, x: torch.Tensor, send_counts: list[int], recv_counts: list[int]
    ) -> torch.Tensor:
        if self.world_size == 1:
            return x

        # 1. Pre-allocate Output Buffer
        # Total tokens I will receive = sum(recv_counts)
        total_recv = sum(recv_counts)
        out_shape = list(x.shape)
        out_shape[0] = total_recv

        output = torch.empty(out_shape, device=x.device, dtype=x.dtype)

        # 2. NCCL All-to-All
        # We use all_to_all_single because it handles the concatenation for us.
        # send_counts and recv_counts act as the "split sizes"
        dist.all_to_all_single(
            output,
            x,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.group,
        )

        return output


class FakeCommunicator(Communicator):
    """For CPU Unit Tests"""

    def exchange_counts(self, local_counts):
        return local_counts

    def all_to_all(self, x, send_counts, recv_counts):
        return x
