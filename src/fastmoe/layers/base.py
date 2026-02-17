import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Function

from fastmoe.comm import Streams


class MoEOverlapFunction(Function):
    @staticmethod
    def forward(ctx, x, block):
        ctx.block = block
        MB = block.n_mb
        chunks = x.chunk(MB, dim=0)
        fwd_ctx = [{} for _ in range(MB)]
        outputs = [None] * MB

        # Events for overlap synchronization
        ev = {s: [torch.cuda.Event() for _ in range(MB)] for s in ["pre", "disp", "exp", "comb"]}

        for tick in range(MB + 4):
            block._fwd_stage_post_ops(tick - 4, fwd_ctx, outputs, ev["comb"])
            block._fwd_stage_experts(tick - 2, fwd_ctx, ev["disp"], ev["exp"])
            block._fwd_stage_pre_ops(tick, fwd_ctx, chunks, ev["pre"])
            block._fwd_stage_combine(tick - 3, fwd_ctx, ev["exp"], ev["comb"])
            block._fwd_stage_dispatch(tick - 1, fwd_ctx, ev["pre"], ev["disp"])

        torch.cuda.current_stream().wait_stream(block.streams[Streams.COMPUTE])
        torch.cuda.current_stream().wait_stream(block.streams[Streams.COMM])
        ctx.fwd_ctx = fwd_ctx
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        block, MB, fwd_ctx = ctx.block, ctx.block.n_mb, ctx.fwd_ctx
        grad_chunks = grad_output.chunk(MB, dim=0)
        ev = {s: [torch.cuda.Event() for _ in range(MB)] for s in ["post", "comb", "exp", "disp"]}
        dx_list = [None] * MB

        for tick in range(MB + 4):
            block._bwd_stage_pre_ops(tick - 4, fwd_ctx, ev["disp"], dx_list)
            block._bwd_stage_experts(tick - 2, fwd_ctx, ev["comb"], ev["exp"])
            block._bwd_stage_post_ops(tick, fwd_ctx, grad_chunks, ev["post"])
            block._bwd_stage_dispatch(tick - 3, fwd_ctx, ev["exp"], ev["disp"])
            block._bwd_stage_combine(tick - 1, fwd_ctx, ev["post"], ev["comb"])

        torch.cuda.current_stream().wait_stream(block.streams[Streams.COMPUTE])
        return torch.cat(dx_list, dim=0), None


class BasePipelineMoE(nn.Module):
    """Base class containing the 5-stage logic. No duplication allowed."""

    def __init__(self, n_mb, group, streams, hidden_dim):
        super().__init__()
        self.n_mb = n_mb
        self.group = group if group else dist.group.WORLD
        self.streams = streams
        self.hidden_dim = hidden_dim
        self.world_size = dist.get_world_size(group=self.group)

        self._metas = None

    def forward(self, x):
        return MoEOverlapFunction.apply(x, self)

    # --- FORWARD STAGES (Verified Logic) ---
    def _fwd_stage_pre_ops(self, mb, ctx, chunks, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        with torch.cuda.stream(stream):
            x = chunks[mb]
            ctx[mb]["x_in"] = x

            # 1. Attention Path (With Metadata Support)
            residual = x
            x_norm = self.input_layernorm(x)

            if self._metas and len(self._metas) > mb:
                # Use metadata if provided (Real Model)
                m = self._metas[mb]

                attn_out = self.self_attn(x_norm, **m)
            else:
                # Fallback / Simple Model
                attn_out = self.self_attn(x_norm)

            x_mid = residual + attn_out

            # 2. MoE Path
            residual_moe = x_mid
            x_norm_moe = self.post_attention_layernorm(x_mid)
            x_flat = x_norm_moe.view(-1, self.hidden_dim)

            # 3. Routing
            perm_in, perm_w, gather_idx, cap = self.gate(x_flat)

            ctx[mb]["residual_moe"] = residual_moe
            ctx[mb]["x_norm_moe"] = x_norm_moe
            ctx[mb]["perm_in"] = perm_in.detach()
            ctx[mb]["perm_w"] = perm_w
            ctx[mb]["gather_idx"] = gather_idx
            ctx[mb]["cap"] = cap
        ev_signal[mb].record(stream)

    def _fwd_stage_dispatch(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            send = ctx[mb]["perm_in"].view(self.world_size, -1, self.hidden_dim)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["dispatched"] = recv.view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _fwd_stage_experts(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            disp = ctx[mb]["dispatched"]
            cap = ctx[mb]["cap"]
            inp = disp.view(self.world_size, self.num_local_experts, cap, self.hidden_dim)
            inp = inp.transpose(0, 1).reshape(self.num_local_experts, -1, self.hidden_dim)
            ctx[mb]["expert_input"] = inp.detach().requires_grad_(True)

            outs = [expert(inp[i]) for i, expert in enumerate(self.local_experts)]
            stack = torch.stack(outs).view(
                self.num_local_experts, self.world_size, cap, self.hidden_dim
            )
            ctx[mb]["expert_out"] = stack.transpose(0, 1).contiguous().view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _fwd_stage_combine(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            send = ctx[mb]["expert_out"].view(self.world_size, -1, self.hidden_dim)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["combined"] = recv.view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _fwd_stage_post_ops(self, mb, ctx, outputs, ev_wait):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            moe_out = ctx[mb]["combined"]
            weighted = moe_out * ctx[mb]["perm_w"].unsqueeze(1)

            buffer = torch.zeros_like(ctx[mb]["residual_moe"].view(-1, self.hidden_dim))
            valid = ctx[mb]["gather_idx"] != -1
            buffer.index_add_(0, ctx[mb]["gather_idx"][valid], weighted[valid])

            shared = self.shared_experts(ctx[mb]["x_norm_moe"])
            outputs[mb] = ctx[mb]["residual_moe"] + buffer.view_as(shared) + shared
            ctx[mb]["moe_out_src"] = moe_out

    # --- BACKWARD STAGES (Autograd Aware) ---
    def _bwd_stage_post_ops(self, mb, ctx, grad_chunks, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        with torch.cuda.stream(stream):
            d_final = grad_chunks[mb]
            ctx[mb]["d_final"] = d_final

            x_norm = ctx[mb]["x_norm_moe"].detach().requires_grad_(True)
            with torch.enable_grad():
                s_out = self.shared_experts(x_norm)
            torch.autograd.backward(s_out, d_final)
            ctx[mb]["d_x_norm_shared"] = x_norm.grad
        ev_signal[mb].record(stream)

    def _bwd_stage_combine(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            d_final = ctx[mb]["d_final"].view(-1, self.hidden_dim)
            d_weighted = torch.zeros_like(ctx[mb]["moe_out_src"])
            valid = ctx[mb]["gather_idx"] != -1
            d_weighted[valid] = d_final[ctx[mb]["gather_idx"][valid]]

            ctx[mb]["d_perm_w"] = (d_weighted * ctx[mb]["moe_out_src"]).sum(dim=1)
            d_moe_out = d_weighted * ctx[mb]["perm_w"].unsqueeze(1)

            send = d_moe_out.view(self.world_size, -1, self.hidden_dim)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["d_expert_out"] = recv.view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _bwd_stage_experts(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            d_out = ctx[mb]["d_expert_out"]
            cap = ctx[mb]["cap"]
            d_out_grp = (
                d_out.view(self.world_size, self.num_local_experts, cap, self.hidden_dim)
                .transpose(0, 1)
                .contiguous()
                .view(self.num_local_experts, -1, self.hidden_dim)
            )

            inp = ctx[mb]["expert_input"]
            with torch.enable_grad():
                outs = [expert(inp[i]) for i, expert in enumerate(self.local_experts)]
                grouped = torch.stack(outs)
            torch.autograd.backward(grouped, d_out_grp)

            d_inp = inp.grad.view(self.num_local_experts, self.world_size, cap, self.hidden_dim)
            ctx[mb]["d_dispatch"] = d_inp.transpose(0, 1).contiguous().view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _bwd_stage_dispatch(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            send = ctx[mb]["d_dispatch"].view(self.world_size, -1, self.hidden_dim)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["d_perm_in"] = recv.view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _bwd_stage_pre_ops(self, mb, ctx, ev_wait, dx_list):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            d_perm_in = ctx[mb]["d_perm_in"]
            d_x_norm_routed = torch.zeros_like(ctx[mb]["x_norm_moe"].view(-1, self.hidden_dim))
            valid = ctx[mb]["gather_idx"] != -1
            d_x_norm_routed.index_add_(0, ctx[mb]["gather_idx"][valid], d_perm_in[valid])

            x_norm = ctx[mb]["x_norm_moe"].detach().requires_grad_(True)
            with torch.enable_grad():
                _, pw, _, _ = self.gate(x_norm.view(-1, self.hidden_dim))
            torch.autograd.backward(pw, ctx[mb]["d_perm_w"])

            d_x_norm = d_x_norm_routed.view_as(x_norm) + x_norm.grad + ctx[mb]["d_x_norm_shared"]

            x_in = ctx[mb]["x_in"].detach().requires_grad_(True)
            d_final = ctx[mb]["d_final"]
            with torch.enable_grad():
                x_n = self.input_layernorm(x_in)

                # Metadata replay
                if self._metas and len(self._metas) > mb:
                    attn = self.self_attn(x_n, **self._metas[mb])
                else:
                    attn = self.self_attn(x_n)

                x_mid = x_in + attn
                x_out_norm = self.post_attention_layernorm(x_mid)

            torch.autograd.backward((x_out_norm, x_mid), (d_x_norm, d_final))
            dx_list[mb] = x_in.grad
