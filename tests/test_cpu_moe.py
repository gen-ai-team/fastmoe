import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

# --- PATCHING (Mock CUDA & Distributed) ---
# We must patch these BEFORE importing the models that might initialize things at module level.
from mocks import MockDist, MockEvent, MockNVTX, MockStream

# Apply patches globally for the test execution
torch.cuda.Event = MockEvent
torch.cuda.Stream = MockStream
torch.cuda.stream = lambda s: s
torch.cuda.current_stream = lambda: MockStream()
torch.cuda.nvtx = MockNVTX
# Mock distributed backend
torch.distributed.all_to_all_single = MockDist.all_to_all_single
torch.distributed.get_world_size = MockDist.get_world_size
torch.distributed.get_rank = MockDist.get_rank
torch.distributed.group = MockDist.group

# ----------------------------

from fastmoe.comm import get_ep_streams  # noqa
from fastmoe.config import MoEScale, get_cfg  # noqa
from fastmoe.models.tiny_model import PipelineMoEBlock, SelfAttention, TinyModel  # noqa


class TestFastMoE(unittest.TestCase):
    def setUp(self):
        """Setup configuration and shared mocks for every test."""
        # 1. Create a standard Tiny Config
        # We simulate world_size=2 to check split logic
        self.cfg = get_cfg(world_size=2, scale=MoEScale.CI)

        # 2. Mock the Stream Dictionary required by the new signature
        # The model expects a dict of {Enum: Stream}
        self.mock_streams = get_ep_streams()

        # 3. Dummy Group (usually a ProcessGroupNCCL, passing a string/int is fine for mocks)
        self.mock_group = "MOCK_GROUP"

    def test_pipeline_block_structure_and_forward(self):
        """
        Verifies that the PipelineMoEBlock instantiates correctly and runs
        a forward pass without crashing on CPU (via mocks).
        """
        print("\n--- [Test] Pipeline Block Logic (CPU Mocked) ---")

        # Dimensions
        B, S, D = self.cfg.moe.batch_size, self.cfg.moe.seqlen, self.cfg.moe.hidden_dim

        # Instantiate Block
        # We test the "B1" style block which has both Pre and Post modules
        block = PipelineMoEBlock(
            cfg=self.cfg,
            group=self.mock_group,
            block_name="TestBlock",
            pre_op_module=SelfAttention(D, self.cfg.moe.num_heads),
            post_op_module=SelfAttention(D, self.cfg.moe.num_heads),
            streams=self.mock_streams,
        )

        # Dummy Input
        x = torch.randn(B, S, D)

        # Execution
        # This exercises the 5-stage loop, split logic, and event recording
        out = block(x)

        # Assertions
        self.assertEqual(out.shape, (B, S, D), "Output shape mismatch")
        self.assertIsInstance(block.pre_ops, SelfAttention)
        self.assertIsInstance(block.post_ops, SelfAttention)
        print("Pipeline forward pass successful.")

    def test_tiny_model_integration(self):
        """
        Verifies the full N-Block TinyModel construction and chain logic.
        """
        print("\n--- [Test] Full TinyModel Integration ---")

        # Patch `get_ep_streams` inside `tiny_model.py` to return our mock dictionary
        # instead of trying to create real CUDA streams inside the model __init__
        with patch("fastmoe.models.tiny_model.get_ep_streams", return_value=self.mock_streams):
            model = TinyModel(cfg=self.cfg, group=self.mock_group)

            # Verify structure
            # Block 0 should have Pre=Attn, Post=Attn
            self.assertIsInstance(model.blocks[0].pre_ops, SelfAttention)
            self.assertIsInstance(model.blocks[0].post_ops, SelfAttention)

            # Block 1 (Last block) should have Pre=Identity (None), Post=Linear
            self.assertIsInstance(model.blocks[1].pre_ops, nn.Identity)
            self.assertIsInstance(model.blocks[1].post_ops, nn.Linear)

            # Forward Pass
            x = torch.randn(self.cfg.moe.batch_size, 10, self.cfg.moe.hidden_dim)
            out = model(x)

            self.assertEqual(out.shape, x.shape)
            print("Full model chain forward pass successful.")

    def test_micro_batch_splitting(self):
        """
        Verifies that chunking logic respects the config.
        """
        print("\n--- [Test] Micro-batch Config Check ---")

        B, S, D = 128, 5, 10
        x = torch.randn(B, S, D)

        # If config says 2 microbatches, we expect chunking to produce tensors of size B/2
        chunks = x.chunk(self.cfg.moe.micro_batches, dim=0)

        self.assertEqual(len(chunks), self.cfg.moe.micro_batches)
        self.assertEqual(chunks[0].shape[0], B // self.cfg.moe.micro_batches)
        print("Micro-batch dimension check passed.")


if __name__ == "__main__":
    unittest.main()
