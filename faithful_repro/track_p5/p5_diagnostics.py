"""P5 numerical diagnostics — instrumentation for a future G5 pilot.
Synthetic fixtures only in G4.1; no HMDD training.

Collects per-step:
- individual loss components + weighted contributions
- parameter-wise update magnitudes (||θ_t - θ_{t-1}||), not just grad norm
- gradient clip before/after statistics
- dynamic graph change (G/ei object changes, edge counts)
- validation checkpoint + threshold provenance
- runtime / memory snapshot
"""
import time, json
import numpy as np
import torch


class Diagnostics:
    def __init__(self):
        self.t0 = time.time()
        self.steps = []
        self.refreshes = []
        self.checkpoints = []
        self.prev_params = None

    def loss_components(self, epoch, comps):
        """comps: dict of raw scalar losses + weighted contributions."""
        rec = {"epoch": int(epoch), "t": round(time.time() - self.t0, 3)}
        rec.update({k: float(v) for k, v in comps.items()})
        self.steps.append(rec)

    def param_update(self, model):
        """Actual per-parameter update magnitude ||Δθ|| — captures what
        optimizer.step() really did (decoupled wd, clipping, adam state)."""
        cur = {n: p.detach().clone() for n, p in model.named_parameters()}
        if self.prev_params is not None:
            deltas = {n: float((cur[n] - self.prev_params[n]).norm())
                      for n in cur}
            self.steps[-1]["param_update_l2_total"] = float(
                np.sqrt(sum(d * d for d in deltas.values())))
            self.steps[-1]["param_update_l2_max"] = float(max(deltas.values()))
            self.steps[-1]["param_update_top"] = max(deltas, key=deltas.get)
        self.prev_params = cur

    def grad_clip_stats(self, epoch, pre_norm, post_norm):
        self.steps[-1].update({
            "grad_norm_preclip": float(pre_norm),
            "grad_norm_postclip": float(post_norm),
            "clip_active": bool(pre_norm > post_norm + 1e-9)})

    def refresh(self, epoch, edge_counts_before, edge_counts_after, G_changed):
        self.refreshes.append({
            "epoch": int(epoch),
            "edge_counts_before": edge_counts_before,
            "edge_counts_after": edge_counts_after,
            "graph_objects_changed": bool(G_changed)})

    def checkpoint(self, epoch, primary_value, threshold_source):
        self.checkpoints.append({"epoch": int(epoch),
                                 "primary_val": float(primary_value),
                                 "threshold_source": threshold_source})

    def report(self):
        return {"steps": self.steps, "refreshes": self.refreshes,
                "checkpoints": self.checkpoints,
                "wall_s": round(time.time() - self.t0, 2),
                "torch_threads": torch.get_num_threads()}


def edge_counts(ei):
    return {str(k): int(v.shape[1]) for k, v in ei.items()}
