import unittest
from unittest.mock import patch

import torch

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
from fastmoe.config import MoEScale, get_ep_cfg  # noqa
from fastmoe.layers.common import Attention  # noqa
from fastmoe.layers.pipeline import PipelineMoELayer  # noqa
from fastmoe.models.tiny_model import TinyDecoderLayer, TinyModel  # noqa


class TestFastMoE(unittest.TestCase):
    def setUp(self):
        """Setup configuration and shared mocks for every test."""
        # 1. Create a standard Tiny Config
        self.cfg = get_ep_cfg(world_size=2, scale=MoEScale.CI)

        # 2. Mock the Stream Dictionary
        # Ensure comm.py returns valid objects (even if Mocks) for CPU testing
        with patch("torch.cuda.is_available", return_value=True):
            self.mock_streams = get_ep_streams()

        # 3. Dummy Group
        self.mock_group = "MOCK_GROUP"

    def test_pipeline_layer_structure_and_forward(self):
        """
        Verifies that PipelineMoELayer correctly wraps a DecoderLayer and runs
        a forward pass without crashing on CPU (via mocks).
        """
        print("\n--- [Test] Pipeline Layer Wrapper (CPU Mocked) ---")

        # Dimensions
        B, S, D = self.cfg.moe.batch_size, self.cfg.moe.seqlen, self.cfg.moe.hidden_dim

        # 1. Create Inner Structure (Real Module)
        inner_layer = TinyDecoderLayer(self.cfg, world_size=2)

        # 2. Wrap it with Pipeline Layer
        ep_layer = PipelineMoELayer(
            layer=inner_layer,
            rank=0,
            world_size=2,
            group=self.mock_group,
            streams=self.mock_streams,
            n_micro_batches=self.cfg.moe.micro_batches,
        )

        # Dummy Input
        x = torch.randn(B, S, D)

        # Execution
        # This exercises the 5-stage loop, permute/unpermute logic, and event recording
        out = ep_layer(x)

        # Assertions
        self.assertEqual(out.shape, (B, S, D), "Output shape mismatch")

        # Verify internal components are mapped correctly
        self.assertIs(ep_layer.self_attn, inner_layer.self_attn)
        self.assertIs(ep_layer.gate, inner_layer.mlp.gate)

        # Verify sharding happened (Local experts should be subset of total)
        total_experts = len(inner_layer.mlp.experts)
        local_experts = len(ep_layer.local_experts)
        self.assertEqual(local_experts, total_experts // 2, "Experts not sharded correctly")

        print("Pipeline Layer forward pass successful.")

    def test_tiny_model_integration(self):
        """
        Verifies the full N-Block TinyModel construction and chain logic.
        """
        print("\n--- [Test] Full TinyModel Integration ---")

        # Patch `get_ep_streams` inside `tiny_model.py`
        with patch("fastmoe.models.tiny_model.get_ep_streams", return_value=self.mock_streams):
            model = TinyModel(cfg=self.cfg, group=self.mock_group)

            # Verify structure
            # Block 0 should be a PipelineMoELayer
            self.assertIsInstance(model.blocks[0], PipelineMoELayer)

            # Verify the inner layer type
            self.assertIsInstance(model.blocks[0].self_attn, Attention)

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
