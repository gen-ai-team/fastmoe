import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Function

from fastmoe.comm import Streams
from fastmoe.models.functional import permute_for_ep, unpermute_from_ep


def create_event() -> torch.cuda.Event | None:
    return torch.cuda.Event() if torch.cuda.is_available() else None


class PipelineMoEFunction(Function):
    @staticmethod
    def forward(ctx, x, block):
        MB = block.n_mb
        metas = block._metas

        # Robust Chunking
        if isinstance(metas, list) and isinstance(metas[0], dict) and "t0" in metas[0]:
            chunks = [x[m["t0"] : m["t1"]] for m in metas]
        else:
            chunks = x.chunk(MB, dim=0)

        fwd = [{} for _ in range(MB)]
        outs = [None] * MB

        ev_pre = [create_event() for _ in range(MB)]
        ev_disp = [create_event() for _ in range(MB)]
        ev_exp = [create_event() for _ in range(MB)]
        ev_comb = [create_event() for _ in range(MB)]

        # --- Forward Tick Loop ---
        for tick in range(MB + 4):
            block._stg5_post_ops(tick - 4, fwd, outs, ev_comb, chunks)
            block._stg4_combine(tick - 3, fwd, ev_exp, ev_comb)
            block._stg3_experts(tick - 2, fwd, ev_disp, ev_exp)
            block._stg2_dispatch(tick - 1, fwd, ev_pre, ev_disp)
            block._stg1_pre_ops(tick, fwd, chunks, ev_pre, metas)

        if torch.cuda.is_available():
            torch.cuda.current_stream().wait_stream(block.streams[Streams.COMPUTE])
            torch.cuda.current_stream().wait_stream(block.streams[Streams.COMM])

        ctx.block = block
        ctx.fwd = fwd
        ctx.chunks = chunks  # Save inputs for backward
        return torch.cat(outs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        block = ctx.block
        fwd = ctx.fwd
        MB = block.n_mb

        # Chunk the incoming gradient to match micro-batches
        # Note: If using variable seqlen (metas), we must slice explicitly using saved metas
        if block._metas and "t0" in block._metas[0]:
            grad_chunks = [grad_output[m["t0"] : m["t1"]] for m in block._metas]
        else:
            grad_chunks = grad_output.chunk(MB, dim=0)

        ev_post = [create_event() for _ in range(MB)]
        ev_comb = [create_event() for _ in range(MB)]
        ev_exp = [create_event() for _ in range(MB)]
        ev_disp = [create_event() for _ in range(MB)]

        dx_list = [None] * MB

        # --- Backward Tick Loop (Reversed) ---
        for tick in range(MB + 4):
            # Pipeline is draining in reverse:
            # Stage 1 (Pre-ops) is now the LAST to run (calculating dx)
            # Stage 5 (Post-ops) is now the FIRST to run (consuming grad_output)

            block._bwd_stg1_pre_ops(tick - 4, fwd, ev_disp, dx_list)
            block._bwd_stg2_dispatch(tick - 3, fwd, ev_exp, ev_disp)
            block._bwd_stg3_experts(tick - 2, fwd, ev_comb, ev_exp)
            block._bwd_stg4_combine(tick - 1, fwd, ev_post, ev_comb)
            block._bwd_stg5_post_ops(tick, fwd, grad_chunks, ev_post)

        if torch.cuda.is_available():
            torch.cuda.current_stream().wait_stream(block.streams[Streams.COMPUTE])

        return torch.cat(dx_list, dim=0), None


class PipelineMoELayer(nn.Module):
    def __init__(self, layer, rank, world_size, group, streams, n_micro_batches=4):
        super().__init__()
        self.rank = rank
        self.ws = world_size
        self.group = group
        self.streams = streams
        self.n_mb = n_micro_batches

        H = layer.input_layernorm.normalized_shape[0]
        self.H = H

        # Modules
        self.input_layernorm = layer.input_layernorm
        self.self_attn = layer.self_attn
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.gate = layer.mlp.gate
        self.shared_experts = layer.mlp.shared_experts

        # Sharded Experts
        all_experts = layer.mlp.experts
        self.ne = len(all_experts)
        self.nl = self.ne // world_size
        s = rank * self.nl
        self.local_experts = nn.ModuleList([all_experts[i] for i in range(s, s + self.nl)])

        self.top_k = layer.cfg.moe.top_k
        self.cap_factor = 4.0
        self._metas = None

    def forward(self, x):
        return PipelineMoEFunction.apply(x, self)

    def _cap(self, N):
        return max(int(math.ceil(N * self.top_k / self.ne * self.cap_factor)), 4)

    # ==================================================================
    # FORWARD STAGES (As verified)
    # ==================================================================

    def _stg1_pre_ops(self, mb, ctx, chunks, ev_signal, metas):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        with torch.cuda.stream(stream):
            x_mb = chunks[mb]
            ctx[mb]["x_in"] = x_mb  # Save for backward

            x_normed = self.input_layernorm(x_mb)
            if isinstance(self.self_attn, nn.Identity):
                attn_out = x_normed
            else:
                attn_out = self.self_attn(x_normed)

            x_after_attn = x_mb + attn_out
            ctx[mb]["x_after_attn"] = x_after_attn  # Save for backward

            x_post_norm = self.post_attention_layernorm(x_after_attn)
            ctx[mb]["x_post_norm"] = x_post_norm  # Save for backward
            x_flat = x_post_norm.view(-1, self.H)

            topk_idx, topk_w = self.gate(x_flat)
            cap = self._cap(x_flat.shape[0])

            perm_in, perm_w, gather_idx = permute_for_ep(x_flat, topk_idx, topk_w, self.ne, cap)

            ctx[mb].update(
                {
                    "perm_in": perm_in.detach(),
                    "perm_w": perm_w,
                    "gather_idx": gather_idx,
                    "cap": cap,
                    "res_moe": x_after_attn.detach(),
                    "shared_in": x_flat.detach(),
                    "N": x_flat.shape[0],
                }
            )
        if ev_signal[mb]:
            ev_signal[mb].record(stream)

    def _stg2_dispatch(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        if ev_wait[mb]:
            stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            cap = ctx[mb]["cap"]
            send = ctx[mb]["perm_in"].view(self.ws, self.nl * cap, self.H)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["dispatched"] = recv.reshape(-1, self.H)
        if ev_signal[mb]:
            ev_signal[mb].record(stream)

    def _stg3_experts(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        if ev_wait[mb]:
            stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            disp = ctx[mb]["dispatched"]
            cap = ctx[mb]["cap"]
            grouped = (
                disp.view(self.ws, self.nl, cap, self.H)
                .transpose(0, 1)
                .reshape(self.nl, -1, self.H)
            )
            ctx[mb]["expert_in"] = grouped.detach().requires_grad_(True)

            outs = [self.local_experts[i](grouped[i]) for i in range(self.nl)]

            exp_out = torch.stack(outs).view(self.nl, self.ws, cap, self.H).transpose(0, 1)
            ctx[mb]["exp_out"] = exp_out.reshape(-1, self.H)
        if ev_signal[mb]:
            ev_signal[mb].record(stream)

    def _stg4_combine(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        if ev_wait[mb]:
            stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            cap = ctx[mb]["cap"]
            send = ctx[mb]["exp_out"].view(self.ws, self.nl * cap, self.H)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["combined"] = recv.reshape(-1, self.H)
        if ev_signal[mb]:
            ev_signal[mb].record(stream)

    def _stg5_post_ops(self, mb, ctx, outputs, ev_wait, chunks):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        if ev_wait[mb]:
            stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            buf = ctx[mb]

            # These return [Batch*Seq, Hidden]
            routed_out = unpermute_from_ep(
                buf["combined"], buf["gather_idx"], buf["perm_w"], buf["N"], self.H
            )
            shared_out = self.shared_experts(buf["shared_in"])

            residual = buf["res_moe"]
            routed_out = routed_out.view_as(residual)
            shared_out = shared_out.view_as(residual)

            outputs[mb] = residual + routed_out + shared_out

            # Save un-reshaped source for backward (it expects flattened)
            ctx[mb]["moe_out_src"] = buf["combined"]

    # ==================================================================
    # BACKWARD STAGES
    # ==================================================================

    def _bwd_stg5_post_ops(self, mb, ctx, grad_chunks, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        with torch.cuda.stream(stream):
            d_final = grad_chunks[mb]
            ctx[mb]["d_final"] = d_final  # Save for Stage 4 & 1

            # 1. Gradient for Shared Experts
            # Reconstruct graph leaf
            x_post_norm = ctx[mb]["x_post_norm"].detach().requires_grad_(True)
            x_flat = x_post_norm.view(-1, self.H)

            # Run Shared Expert forward again to connect autograd
            with torch.enable_grad():
                s_out = self.shared_experts(x_flat)
            torch.autograd.backward(s_out, d_final)
            ctx[mb]["d_x_post_norm_shared"] = x_post_norm.grad

        if ev_signal[mb]:
            ev_signal[mb].record(stream)

    def _bwd_stg4_combine(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        if ev_wait[mb]:
            stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            d_final = ctx[mb]["d_final"].view(-1, self.H)
            moe_out_src = ctx[mb]["moe_out_src"]

            # Reverse Unpermute (Scatter -> Gather)
            # Forward: buffer.index_add_(gather_idx, output * perm_w)
            # Backward: d_output = d_buffer[gather_idx] * perm_w

            valid = ctx[mb]["gather_idx"] != -1
            gather_idx = ctx[mb]["gather_idx"][valid]

            d_weighted = torch.zeros_like(moe_out_src)
            d_weighted[valid] = d_final[gather_idx]

            # Gradient wrt perm_w (TopK Weights)
            # d_perm_w = sum(d_buffer[gather_idx] * output)
            ctx[mb]["d_perm_w"] = (d_weighted * moe_out_src).sum(dim=1)

            # Gradient wrt Expert Output
            d_moe_out = d_weighted * ctx[mb]["perm_w"].unsqueeze(1)

            # Reverse AllToAll
            cap = ctx[mb]["cap"]
            send = d_moe_out.view(self.ws, self.nl * cap, self.H)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["d_expert_out"] = recv.reshape(-1, self.H)

        if ev_signal[mb]:
            ev_signal[mb].record(stream)

    def _bwd_stg3_experts(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        if ev_wait[mb]:
            stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            d_out = ctx[mb]["d_expert_out"]
            cap = ctx[mb]["cap"]

            # Reshape to group by local expert
            d_out_grp = (
                d_out.view(self.ws, self.nl, cap, self.H)
                .transpose(0, 1)
                .reshape(self.nl, -1, self.H)
            )

            inp = ctx[mb]["expert_in"]

            # Trigger Autograd for Local Experts
            with torch.enable_grad():
                outs = [self.local_experts[i](inp[i]) for i in range(self.nl)]
                grouped = torch.stack(outs)

            # d_out_grp matches shape of grouped [nl, ws*cap, H]
            torch.autograd.backward(grouped, d_out_grp)

            d_inp = inp.grad
            # Inverse Transpose/View
            d_inp = d_inp.view(self.nl, self.ws, cap, self.H).transpose(0, 1)
            ctx[mb]["d_dispatch"] = d_inp.reshape(-1, self.H)

        if ev_signal[mb]:
            ev_signal[mb].record(stream)

    def _bwd_stg2_dispatch(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        if ev_wait[mb]:
            stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            cap = ctx[mb]["cap"]
            send = ctx[mb]["d_dispatch"].view(self.ws, self.nl * cap, self.H)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["d_perm_in"] = recv.reshape(-1, self.H)

        if ev_signal[mb]:
            ev_signal[mb].record(stream)

    def _bwd_stg1_pre_ops(self, mb, ctx, ev_wait, dx_list):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        if ev_wait[mb]:
            stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            d_perm_in = ctx[mb]["d_perm_in"]

            # Reverse Permute (Gather -> Scatter)
            # Forward: perm_in[i] = x_flat[gather_idx[i]]
            # Backward: d_x_flat.index_add_(gather_idx, d_perm_in)

            d_x_routed = torch.zeros_like(ctx[mb]["shared_in"])  # [N, H]
            valid = ctx[mb]["gather_idx"] != -1
            gather_idx = ctx[mb]["gather_idx"][valid]

            d_x_routed.index_add_(0, gather_idx, d_perm_in[valid])

            # Backward for Gate (TopK Router)
            x_post_norm = ctx[mb]["x_post_norm"].detach().requires_grad_(True)
            x_flat = x_post_norm.view(-1, self.H)

            with torch.enable_grad():
                # Re-run gate to get graph
                _, pw = self.gate(x_flat)

            # Propagate d_perm_w into Gate
            # Note: We only propagate gradients into the weights (pw), not the indices
            torch.autograd.backward(pw, ctx[mb]["d_perm_w"])
            d_x_gate = x_post_norm.grad

            # Total Gradient at Post-Attention LN output
            d_x_post_norm = (
                d_x_routed.view_as(x_post_norm) + d_x_gate + ctx[mb]["d_x_post_norm_shared"]
            )

            # Backward for Post-Attention LN & Pre-Ops
            x_in = ctx[mb]["x_in"].detach().requires_grad_(True)
            x_after_attn = ctx[mb]["x_after_attn"].detach().requires_grad_(True)
            d_final = ctx[mb]["d_final"]  # Residual 2 gradient

            with torch.enable_grad():
                x_post = self.post_attention_layernorm(x_after_attn)

                # Reconstruct Pre-Ops
                x_normed = self.input_layernorm(x_in)
                if isinstance(self.self_attn, nn.Identity):
                    attn = x_normed
                else:
                    attn = self.self_attn(x_normed)
                x_mid = x_in + attn

            # Gradients flowing back:
            # 1. d_x_post_norm flows into x_post -> x_after_attn
            # 2. d_final (Residual 2) flows into x_after_attn
            d_x_after_attn_total = (
                torch.autograd.grad(x_post, x_after_attn, d_x_post_norm, retain_graph=True)[0]
                + d_final
            )

            # 3. d_x_after_attn_total flows into x_mid -> x_in
            dx = torch.autograd.grad(x_mid, x_in, d_x_after_attn_total)[0]

            dx_list[mb] = dx
