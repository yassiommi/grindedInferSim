"""Per-layer execution time tracking with GPU-initiated KV cache prefetching.

Records compute, I/O, and communication times for each transformer layer,
supports prefetch overlap simulation where the GPU initiates KV cache
loading for the next layer while the current layer is computing.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LayerTiming:
    """Timing breakdown for a single transformer layer."""
    layer_index: int

    # Compute (seconds)
    attn_core_time: float = 0.0
    attn_proj_time: float = 0.0  # QKV/O projections
    moe_compute_time: float = 0.0  # Routed + shared experts
    shared_expert_time: float = 0.0

    # I/O (seconds)
    kv_cache_load_time: float = 0.0
    expert_weight_load_time: float = 0.0

    # Communication (seconds)
    comm_before_moe: float = 0.0  # All-reduce or dispatch
    comm_after_moe: float = 0.0  # All-reduce or combine

    # Prefetch
    prefetch_overlap: float = 0.0  # Time saved by prefetching

    # Flags
    is_moe: bool = False

    @property
    def compute_time(self) -> float:
        return self.attn_core_time + self.attn_proj_time + self.moe_compute_time + self.shared_expert_time

    @property
    def io_time(self) -> float:
        return self.kv_cache_load_time + self.expert_weight_load_time

    @property
    def comm_time(self) -> float:
        return self.comm_before_moe + self.comm_after_moe

    @property
    def effective_io_time(self) -> float:
        return max(0.0, self.io_time - self.prefetch_overlap)

    @property
    def total_time(self) -> float:
        return self.compute_time + self.effective_io_time + self.comm_time

    @property
    def total_time_no_overlap(self) -> float:
        return self.compute_time + self.io_time + self.comm_time

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "attn_core_time_us": self.attn_core_time * 1e6,
            "attn_proj_time_us": self.attn_proj_time * 1e6,
            "moe_compute_time_us": self.moe_compute_time * 1e6,
            "shared_expert_time_us": self.shared_expert_time * 1e6,
            "kv_cache_load_time_us": self.kv_cache_load_time * 1e6,
            "expert_weight_load_time_us": self.expert_weight_load_time * 1e6,
            "comm_before_moe_us": self.comm_before_moe * 1e6,
            "comm_after_moe_us": self.comm_after_moe * 1e6,
            "prefetch_overlap_us": self.prefetch_overlap * 1e6,
            "compute_time_us": self.compute_time * 1e6,
            "io_time_us": self.io_time * 1e6,
            "effective_io_time_us": self.effective_io_time * 1e6,
            "comm_time_us": self.comm_time * 1e6,
            "total_time_us": self.total_time * 1e6,
            "is_moe": self.is_moe,
        }


def apply_kv_prefetch(layers: List[LayerTiming]) -> float:
    """Apply GPU-initiated KV cache prefetching across layers.

    At the start of layer N, a prefetch of KV cache for layer N+1 is
    initiated on a separate DMA engine. Layer N+1 cannot start until
    all I/O and comm from layer N is complete, but the KV load can
    overlap with layer N's compute phase.

    Returns total savings in seconds.
    """
    total_savings = 0.0
    for i in range(len(layers) - 1):
        current = layers[i]
        next_layer = layers[i + 1]
        # Overlap = min(current compute, next KV load)
        overlap = min(current.compute_time, next_layer.kv_cache_load_time)
        next_layer.prefetch_overlap = overlap
        total_savings += overlap
    return total_savings


def save_layer_timings(layers: List[LayerTiming], output_dir: str, phase: str = "decode"):
    """Save layer timing data and generate Gantt plot."""
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    data = [l.to_dict() for l in layers]
    json_path = os.path.join(output_dir, f"layer_timings_{phase}.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Layer timings saved to {json_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, f"layer_timings_{phase}.csv")
    if data:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"Layer timings CSV saved to {csv_path}")


def plot_layer_gantt(layers: List[LayerTiming], output_dir: str,
                     phase: str = "decode", enable_prefetch: bool = False):
    """Generate Gantt-style plot of per-layer timing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping Gantt plot")
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, max(6, len(layers) * 0.35)))
    bar_height = 0.6

    for i, layer in enumerate(layers):
        y = len(layers) - 1 - i
        offset = 0.0

        # Attention compute (green)
        attn = (layer.attn_core_time + layer.attn_proj_time) * 1e6  # us
        if attn > 0:
            ax.barh(y, attn, left=offset, height=bar_height,
                    color="#4CAF50", edgecolor="black", linewidth=0.3)
            offset += attn

        # MoE/FFN compute (lime)
        moe = (layer.moe_compute_time + layer.shared_expert_time) * 1e6
        if moe > 0:
            color = "#8BC34A" if layer.is_moe else "#4CAF50"
            ax.barh(y, moe, left=offset, height=bar_height,
                    color=color, edgecolor="black", linewidth=0.3)
            offset += moe

        # I/O (blue)
        io = layer.effective_io_time * 1e6
        if io > 0:
            ax.barh(y, io, left=offset, height=bar_height,
                    color="#2196F3", edgecolor="black", linewidth=0.3)
            offset += io

        # Communication (orange)
        comm = layer.comm_time * 1e6
        if comm > 0:
            ax.barh(y, comm, left=offset, height=bar_height,
                    color="#FF9800", edgecolor="black", linewidth=0.3)
            offset += comm

        # Prefetch savings indicator
        if layer.prefetch_overlap > 0:
            savings = layer.prefetch_overlap * 1e6
            ax.barh(y, savings, left=offset, height=bar_height * 0.3,
                    color="#F44336", alpha=0.5, hatch="//",
                    edgecolor="red", linewidth=0.3)

        if layer.is_moe:
            ax.annotate("MoE", xy=(1, y), fontsize=5, color="purple", fontweight="bold")

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l.layer_index}" for l in reversed(layers)], fontsize=7)
    ax.set_xlabel("Time (us)")

    title = f"Per-Layer {phase.capitalize()} Execution"
    if enable_prefetch:
        total_savings = sum(l.prefetch_overlap for l in layers) * 1e6
        title += f" (Prefetch saves {total_savings:.0f}us)"
    ax.set_title(title)

    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Attention"),
        mpatches.Patch(color="#8BC34A", label="MoE/FFN"),
        mpatches.Patch(color="#2196F3", label="I/O (effective)"),
        mpatches.Patch(color="#FF9800", label="Communication"),
    ]
    if enable_prefetch:
        legend_patches.append(
            mpatches.Patch(facecolor="#F44336", alpha=0.5, hatch="//",
                          edgecolor="red", label="Prefetch savings")
        )
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"layer_gantt_{phase}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Gantt plot saved to {plot_path}")
