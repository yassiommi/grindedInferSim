# Layer-Level Timing and GPU-Initiated KV Cache Prefetching

This document describes extensions to InferSim that add per-layer execution time breakdowns with Gantt-style visualization and GPU-initiated KV cache prefetching simulation.

## Table of Contents

- [Overview](#overview)
- [Per-Layer Timing](#per-layer-timing)
  - [What is Tracked](#what-is-tracked)
  - [LayerTiming Dataclass](#layertiming-dataclass)
- [GPU-Initiated KV Cache Prefetching](#gpu-initiated-kv-cache-prefetching)
  - [Background](#background)
  - [How It Works](#how-it-works)
  - [When Prefetching Helps](#when-prefetching-helps)
- [Gantt-Style Visualization](#gantt-style-visualization)
- [MoE and Sparse Model Support](#moe-and-sparse-model-support)
  - [Existing MoE Support](#existing-moe-support)
  - [DeepSeek-V3 Sparse Attention (MLA)](#deepseek-v3-sparse-attention-mla)
  - [Expert Weight Loading vs Compute](#expert-weight-loading-vs-compute)
- [Usage](#usage)
  - [New CLI Arguments](#new-cli-arguments)
  - [Examples](#examples)
- [Output Files](#output-files)
- [Architecture](#architecture)

---

## Overview

InferSim predicts LLM inference performance using FLOPs-based modeling with empirical MFU (Model FLOPs Utilization) benchmarks. Previously, all layers were assumed identical and their times aggregated as:

```
TPOT = num_layers * (attn_time + moe_time + comm_time) + overhead
```

This extension adds:
1. **Per-layer timing**: Each layer gets its own `LayerTiming` with compute, I/O, and communication breakdowns
2. **KV cache prefetching**: Simulates GPU-initiated DMA transfers that overlap next-layer KV cache loading with current-layer computation
3. **Gantt visualization**: Horizontal bar charts showing the time composition of each layer

---

## Per-Layer Timing

### What is Tracked

For each of the `num_hidden_layers` layers, the following times are recorded (in seconds):

| Field | Description | Bottleneck |
|-------|-------------|------------|
| `attn_core_time` | Attention QK^T and softmax-V computation | Compute or I/O (KV load) |
| `attn_proj_time` | QKV/O projection GEMMs (or MLA up/down proj) | Compute |
| `moe_compute_time` | Routed expert grouped GEMM | Compute |
| `shared_expert_time` | Shared expert dense GEMM | Compute |
| `kv_cache_load_time` | Loading KV cache from HBM | I/O (memory BW) |
| `expert_weight_load_time` | Loading expert weights from HBM | I/O (memory BW) |
| `comm_before_moe` | All-reduce or dispatch communication | Communication |
| `comm_after_moe` | All-reduce or combine communication | Communication |
| `prefetch_overlap` | Time saved by KV prefetching | Optimization |

### LayerTiming Dataclass

```python
@dataclass
class LayerTiming:
    layer_index: int
    attn_core_time: float = 0.0      # seconds
    attn_proj_time: float = 0.0
    moe_compute_time: float = 0.0
    shared_expert_time: float = 0.0
    kv_cache_load_time: float = 0.0
    expert_weight_load_time: float = 0.0
    comm_before_moe: float = 0.0
    comm_after_moe: float = 0.0
    prefetch_overlap: float = 0.0
    is_moe: bool = False

    @property
    def compute_time(self) -> float: ...     # All compute
    @property
    def io_time(self) -> float: ...          # All I/O
    @property
    def effective_io_time(self) -> float: ... # I/O minus prefetch savings
    @property
    def total_time(self) -> float: ...       # compute + effective_io + comm
```

---

## GPU-Initiated KV Cache Prefetching

### Background

During decode, each attention layer must load the KV cache for all previous tokens from HBM (GPU global memory). For long sequences and large batch sizes, this KV cache loading becomes the dominant cost:

```
KV load time per layer = kvcache_bytes_per_token * context_length * batch_size / memory_bandwidth
```

For example, with DeepSeek-V3 (MLA, kv_lora_rank=512):
- KV cache per token per layer: ~1.2 KB (FP8)
- Context 4K, batch 64: ~300 MB per layer
- H800 bandwidth (2744 GB/s * 0.8): ~137 us per layer

Modern GPUs have independent DMA engines that can transfer data concurrently with compute. GPU-initiated prefetching exploits this by starting the KV cache load for layer N+1 during layer N's computation phase.

### How It Works

```
apply_kv_prefetch(layers):
    for i in range(len(layers) - 1):
        current = layers[i]
        next_layer = layers[i + 1]
        overlap = min(current.compute_time, next_layer.kv_cache_load_time)
        next_layer.prefetch_overlap = overlap
```

**Constraints:**
- Layer N+1 cannot start computing until layer N's I/O and comm are complete
- The DMA prefetch runs concurrently with layer N's compute
- Savings = min(compute_time_layer_N, kv_load_time_layer_N+1)
- First layer gets no savings (no previous layer to overlap with)
- Last layer's compute enables no savings (no next layer)

### When Prefetching Helps

Prefetching is most beneficial when:
- **Decode phase** (not prefill - prefill generates KV, doesn't load it)
- **Long sequences**: More KV cache to load per layer
- **Large batch sizes**: More total KV bytes
- **I/O-bound layers**: When `kv_load_time > compute_time`, prefetch can hide most of the I/O
- **MoE models**: Expert weight loading can also be a bottleneck

Prefetching has diminishing returns when layers are compute-bound (compute >> I/O).

---

## Gantt-Style Visualization

The plotter generates horizontal bar charts where each layer is a row:

```
Layer 0: [====Attention====][==MoE==][=I/O=][Comm]
Layer 1: [====Attention====][==MoE==][Comm]        <- I/O prefetched
Layer 2: [====Attention====][==MoE==][Comm]        <- I/O prefetched
...
```

**Color scheme:**
- Green (#4CAF50): Attention compute
- Lime (#8BC34A): MoE/FFN compute
- Blue (#2196F3): I/O (effective, after prefetch savings)
- Orange (#FF9800): Communication
- Red hatching: Prefetch savings

---

## MoE and Sparse Model Support

### Existing MoE Support

InferSim already supports MoE models through:
- `layers/moe.py`: `MoE` class computing routed and shared expert latencies
- `mfu/mfu.py`: Grouped GEMM MFU benchmarks from actual GPU runs
- `comm/comm.py`: DeepEP dispatch/combine communication
- `config/model_config.py`: Config parsing for `num_routed_experts`, `num_experts_per_tok`, etc.

### DeepSeek-V3 Sparse Attention (MLA)

DeepSeek-V3 uses Multi-head Latent Attention (MLA) which compresses KV representations:

```
Standard MHA KV cache: 2 * num_kv_heads * head_dim per token per layer
MLA KV cache: kv_lora_rank + qk_rope_head_dim per token per layer (much smaller)
```

The `layers/attn.py` MLA class handles:
- Compressed KV down-projection: `hidden -> kv_lora_rank`
- Absorbed attention: Fuses K-up and V-up projections into attention weights
- Q LoRA: `hidden -> q_lora_rank -> num_heads * qk_head_dim`

### Expert Weight Loading vs Compute

For MoE layers, the time is:
```
moe_time = max(expert_compute_time, expert_weight_load_time) + shared_expert_time
```

This `max()` captures that the memory controller loads weights while tensor cores compute. The layer timing now tracks both separately, enabling analysis of whether each layer is compute-bound or I/O-bound.

---

## Usage

### New CLI Arguments

```
--enable-prefetch       Enable GPU-initiated KV cache prefetching
--output-dir DIR        Output directory for timing data and plots (default: ./output)
```

### Examples

**DeepSeek-V3 with prefetching (128 GPUs):**
```bash
python3 main.py \
    --config-path hf_configs/deepseek_v3_config.json \
    --device-type H800 \
    --world-size 128 --num-nodes 16 \
    --use-fp8-gemm --enable-deepep --enable-tbo \
    --target-osl 1786 --decode-bs 64 \
    --enable-prefetch \
    --output-dir ./output/deepseek_v3_prefetch \
    --decode-only
```

**Qwen3-30B with layer timing:**
```bash
python3 main.py \
    --config-path hf_configs/qwen3-30B-A3B_config.json \
    --device-type H800 \
    --world-size 8 --num-nodes 1 \
    --enable-prefetch \
    --output-dir ./output/qwen3_30b
```

**Dense model (Llama-style) with prefetching:**
```bash
python3 main.py \
    --config-path hf_configs/your_dense_config.json \
    --device-type H200 \
    --world-size 1 \
    --enable-prefetch \
    --output-dir ./output/dense_prefetch
```

---

## Output Files

When running with `--output-dir`, the following are generated:

```
output/
├── layer_timings_prefill.json    # Per-layer timing data (prefill phase)
├── layer_timings_prefill.csv     # CSV format for analysis
├── layer_timings_decode.json     # Per-layer timing data (decode phase)
├── layer_timings_decode.csv      # CSV format for analysis
├── layer_gantt_prefill.png       # Gantt chart (prefill)
└── layer_gantt_decode.png        # Gantt chart (decode)
```

The CSV files contain one row per layer with columns:
`layer_index, attn_core_time_us, attn_proj_time_us, moe_compute_time_us, kv_cache_load_time_us, expert_weight_load_time_us, comm_before_moe_us, comm_after_moe_us, prefetch_overlap_us, compute_time_us, io_time_us, effective_io_time_us, comm_time_us, total_time_us, is_moe`

---

## Architecture

### New Files
- `layers/layer_timing.py` - `LayerTiming` dataclass, `apply_kv_prefetch()`, Gantt plotter, data export

### Modified Files
- `models/model.py` - Per-layer timing construction, prefetch integration, breakdown printing
- `main.py` - New CLI arguments (`--enable-prefetch`, `--output-dir`)

### Data Flow

```
Model.decoding()
    |
    |-- Compute per-layer attention/MoE/comm times (using existing layer classes)
    |-- Create LayerTiming for each layer
    |-- If enable_prefetch: apply_kv_prefetch(layer_timings)
    |-- Adjust TPOT by prefetch savings
    |-- Print per-layer breakdown table
    |-- save_layer_timings() -> JSON + CSV
    |-- plot_layer_gantt() -> PNG
```

The existing `MoE`, `MHA`/`MLA`, and `Comm` classes are used unchanged - the new code wraps their outputs into per-layer `LayerTiming` objects for tracking and visualization.
