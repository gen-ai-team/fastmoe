import torch
import torch.nn as nn
from torch.profiler import record_function


class PipelinedMoEBlock(nn.Module):
    def __init__(self, block_module, comm_stream):
        super().__init__()
        # Share layers from the original block
        self.attn = block_module.attn
        self.norm1 = block_module.norm1
        self.norm2 = block_module.norm2
        self.moe = block_module.ff
        self.comm_stream = comm_stream

    def forward(self, x):
        """
        Memory-Optimized Hybrid Pipeline with Non-Blocking CPU Dispatch
        """
        # --- PHASE 1: STANDARD ATTENTION ---
        # Keep full batch for max tensor core utilization
        with record_function("Standard: Attention"):
            h = self.norm1(x)
            h = self.attn(h)
            x = x + h

        # --- PHASE 2: PIPELINED MOE ---
        # 1. Pre-calculate Norm
        moe_input = self.norm2(x)

        # 2. Split EVERYTHING to allow early freeing
        # Splitting 'x' (the residual) allows us to add it incrementally
        # and drop the reference to the massive full tensor.
        x_chunks = x.chunk(2, dim=0)
        c1_resid, c2_resid = x_chunks[0], x_chunks[1]

        chunks = moe_input.chunk(2, dim=0)
        c1, c2 = chunks[0], chunks[1]

        # Delete full references to allow garbage collection if needed
        del x, moe_input, x_chunks

        # Metadata & Events
        c1_meta = {}
        c2_meta = {}
        ev_dispatch_c1 = torch.cuda.Event()
        ev_compute_c1 = torch.cuda.Event()
        ev_dispatch_c2 = torch.cuda.Event()

        # --- STEP 1: Process C1 (Dispatch) ---
        with record_function("Pipe: Gate C1"):
            perm1, sc1, rev1, w1, s1 = self.moe.gate_and_sort(c1)
            c1_meta = {
                "rev": rev1,
                "w": w1,
                "s": s1,
            }

        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with record_function("Pipe: Dispatch C1"):
                rd1, rc1, sl1 = self.moe.dispatch_exchange(perm1, sc1)
                c1_meta["rd"] = rd1
                c1_meta["rc"] = rc1
                c1_meta["sl"] = sl1
            ev_dispatch_c1.record()

        # --- STEP 2: Process C2 & Compute C1 ---
        with record_function("Pipe: Gate C2"):
            perm2, sc2, rev2, w2, s2 = self.moe.gate_and_sort(c2)
            c2_meta = {"rev": rev2, "w": w2, "s": s2}

        # Wait for Dispatch C1 to finish before computing
        torch.cuda.current_stream().wait_event(ev_dispatch_c1)

        with record_function("Pipe: Experts C1"):
            eo1 = self.moe.compute_experts(c1_meta["rd"], c1_meta["rc"])
            # Free the heavy input buffer immediately
            del c1_meta["rd"]
            c1_meta["eo"] = eo1
        ev_compute_c1.record()

        # --- STEP 3: Dispatch C2 & Combine C1 ---
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with record_function("Pipe: Dispatch C2"):
                rd2, rc2, sl2 = self.moe.dispatch_exchange(perm2, sc2)
                c2_meta["rd"] = rd2
                c2_meta["rc"] = rc2
                c2_meta["sl"] = sl2
            ev_dispatch_c2.record()

            # Wait for Experts C1 to finish before combining
            self.comm_stream.wait_event(ev_compute_c1)

            with record_function("Pipe: Combine C1"):
                fd1 = self.moe.combine_exchange(c1_meta["eo"], c1_meta["rc"], c1_meta["sl"])
                del c1_meta["eo"], c1_meta["rc"], c1_meta["sl"]

                # Unpermute
                res1 = self.moe.unpermute(fd1, c1_meta["rev"], c1_meta["w"], c1_meta["s"])
                del fd1, c1_meta

        # --- STEP 4: Compute C2 ---
        torch.cuda.current_stream().wait_event(ev_dispatch_c2)

        with record_function("Pipe: Experts C2"):
            eo2 = self.moe.compute_experts(c2_meta["rd"], c2_meta["rc"])
            del c2_meta["rd"]

        # --- STEP 5: Combine C2 ---
        torch.cuda.current_stream().wait_stream(self.comm_stream)

        with record_function("Pipe: Combine C2"):
            fd2 = self.moe.combine_exchange(eo2, c2_meta["rc"], c2_meta["sl"])
            res2 = self.moe.unpermute(fd2, c2_meta["rev"], c2_meta["w"], c2_meta["s"])
            del fd2, c2_meta, eo2

        # --- Incremental Add ---
        # Add residuals to each chunk.
        out1 = res1 + c1_resid
        out2 = res2 + c2_resid

        return torch.cat([out1, out2], dim=0)
