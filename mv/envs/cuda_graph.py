"""Replay a function as a CUDA graph: record its GPU work once, then re-run it with one call.

Why (engineering log WP63): one simulation step for 15 000 cars is a few hundred tiny GPU operations
(adding two lists of 15 000 numbers takes the GPU microseconds).  Each operation costs about as much to
*launch* from Python as to run, and the GPU waits for the CPU in between.  A CUDA graph records the whole
sequence once; replaying it launches everything in one go.  Same kernels, same order, so the results are
bit-for-bit identical to running the function normally (tested).

The catch: a graph reads and writes fixed memory addresses.  Our code often *replaces* a state tensor
(``self.vx = self.vx + ...``) instead of overwriting it.  This helper handles that automatically: every
tensor attribute of the given objects that the function replaces is copied back into the original
tensor at the end of the recorded work, so the next replay reads the new state from the same address.
Rules for the recorded function: no copying values to the CPU (no ``int(tensor)``, ``.item()``,
``nonzero()``), no random numbers, and tensor shapes must not change from call to call.
"""

from __future__ import annotations

import torch

__all__ = ["GraphedFunction"]


class GraphedFunction:
    def __init__(self, fn, owners: list, warmup: int = 2):
        self.fn, self.owners, self.warmup = fn, owners, warmup
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_in: list[torch.Tensor] = []
        self.static_out: tuple[torch.Tensor, ...] = ()

    def _tensor_attrs(self) -> dict[tuple[int, str], torch.Tensor]:
        return {(i, k): v for i, o in enumerate(self.owners) for k, v in vars(o).items() if isinstance(v, torch.Tensor)}

    def _restore(self, attrs: dict, snap: dict) -> None:
        for (i, k), v0 in attrs.items():
            vars(self.owners[i])[k] = v0
            v0.copy_(snap[(i, k)])

    def _capture(self, inputs: tuple[torch.Tensor, ...]) -> None:
        attrs = self._tensor_attrs()
        snap = {key: v.clone() for key, v in attrs.items()}
        self.static_in = [x.clone() for x in inputs]
        # warm-up on a side stream (PyTorch's recipe: lets libraries set themselves up before recording);
        # it changes the state, which is then put back exactly as it was
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self.fn(*self.static_in)
        torch.cuda.current_stream().wait_stream(side)
        self._restore(attrs, snap)
        del snap
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.fn(*self.static_in)
            for (i, k), v0 in attrs.items():          # replaced state -> back into its original tensor
                v1 = vars(self.owners[i])[k]
                if v1 is not v0:
                    v0.copy_(v1)
                    vars(self.owners[i])[k] = v0
        self.graph, self.static_out = g, tuple(out)

    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.graph is None:
            self._capture(inputs)
        for s, x in zip(self.static_in, inputs):
            s.copy_(x)
        self.graph.replay()
        # copies: the graph overwrites its outputs at the next replay, callers may keep these longer
        return tuple(o.clone() for o in self.static_out)
