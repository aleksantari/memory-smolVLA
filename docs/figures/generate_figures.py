"""Generate figures for v5 writeup PDF.

Run from repo root:
    python docs/figures/generate_figures.py

Outputs PNG figures to docs/figures/.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FIG_DIR = Path(__file__).parent

# --------------------------------------------------------------------
# Style
# --------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_FROZEN = "#cfd8dc"
C_TRAIN = "#ffe082"
C_MEMORY = "#90caf9"
C_DATA = "#a5d6a7"
C_ACCENT = "#ef5350"
C_NEUTRAL = "#eceff1"


def box(ax, x, y, w, h, text, color=C_NEUTRAL, fontsize=9, ec="black"):
    """Draw a rounded rectangle with centered text."""
    bbox = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.0, edgecolor=ec, facecolor=color,
    )
    ax.add_patch(bbox)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, x1, y1, x2, y2, label=None, color="black", lw=1.4):
    """Draw an arrow with optional label."""
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=14,
        linewidth=lw, color=color,
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, label,
                ha="center", va="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


# --------------------------------------------------------------------
# Figure 1 — SmolVLA architecture
# --------------------------------------------------------------------
def fig_smolvla():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9)
    ax.axis("off")

    # Inputs
    box(ax, 0.2, 7.8, 2.2, 0.9, "RGB image\n(camera 1)", C_DATA)
    box(ax, 0.2, 6.6, 2.2, 0.9, "RGB image\n(camera 2)", C_DATA)
    box(ax, 0.2, 5.4, 2.2, 0.9, "Language\ninstruction", C_DATA)
    box(ax, 0.2, 4.2, 2.2, 0.9, "Robot state\n(8-dim)", C_DATA)

    # Vision encoder
    box(ax, 3.0, 7.0, 2.5, 1.7,
        "SigLIP\n+ PixelShuffle\n→ 64 tokens/img", C_FROZEN)
    arrow(ax, 2.4, 8.25, 3.0, 7.85)
    arrow(ax, 2.4, 7.05, 3.0, 7.5)

    # Language tokenizer
    box(ax, 3.0, 5.4, 2.5, 0.9, "Tokenizer\n→ ~30–50 tokens", C_FROZEN)
    arrow(ax, 2.4, 5.85, 3.0, 5.85)

    # State proj
    box(ax, 3.0, 4.2, 2.5, 0.9, "Linear proj\n→ 1 token (576-d)", C_FROZEN)
    arrow(ax, 2.4, 4.65, 3.0, 4.65)

    # Concat → prefix
    box(ax, 6.1, 5.0, 2.5, 3.0,
        "Prefix\n[image_start, 64×2 imgs,\nimage_end, lang, state]\n\n~163–183 tokens × 576",
        C_NEUTRAL, fontsize=8.5)
    arrow(ax, 5.5, 7.85, 6.1, 7.0)
    arrow(ax, 5.5, 5.85, 6.1, 6.5)
    arrow(ax, 5.5, 4.65, 6.1, 5.5)

    # SmolLM2 truncated
    box(ax, 9.2, 4.5, 2.0, 4.0,
        "SmolLM2-360M\n\nLayers 0–15\nUSED\n\n(visual peak)\n\nLayers 16–31\nDISCARDED\n(semantic)",
        C_FROZEN, fontsize=8)
    # Show truncation visually
    ax.add_patch(Rectangle((9.2, 4.5), 2.0, 1.6, facecolor="#bdbdbd",
                           edgecolor="black", alpha=0.6))
    ax.text(10.2, 5.3, "discarded", ha="center", va="center",
            fontsize=8, color="#666", style="italic")

    arrow(ax, 8.6, 6.5, 9.2, 6.5)

    # Action expert
    box(ax, 11.6, 5.6, 2.2, 1.6,
        "Action expert\n(trainable in V4)\n\nFlow matching\n10 denoise steps", C_TRAIN, fontsize=8.5)
    arrow(ax, 11.2, 7.0, 11.6, 6.8, label="KV cache")

    # Output
    box(ax, 11.6, 3.6, 2.2, 0.9, "Action chunk\n[B, 50, 8]", C_DATA)
    arrow(ax, 12.7, 5.6, 12.7, 4.5)

    # Title and annotations
    ax.text(7, 8.7, "SmolVLA — base architecture",
            ha="center", fontsize=14, weight="bold")
    ax.text(7, 0.7, "Note: SmolVLA truncates SmolLM2 at 16/32 layers. "
            "At our deepest available point (layer 15), hidden states are "
            "predominantly perceptual — no deep semantic stream available.",
            ha="center", fontsize=8.5, style="italic", color="#444")
    ax.text(7, 0.2, "Frozen modules in gray. "
            "Only the action expert + state_proj are trainable in baseline.",
            ha="center", fontsize=8, color="#666")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_smolvla.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig01_smolvla.png")


# --------------------------------------------------------------------
# Figure 2 — MemoryVLA architecture
# --------------------------------------------------------------------
def fig_memoryvla():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9)
    ax.axis("off")

    # Inputs
    box(ax, 0.2, 6.5, 2.0, 1.5, "RGB image", C_DATA)
    box(ax, 0.2, 4.8, 2.0, 0.9, "Language", C_DATA)

    # Dual vision encoders
    box(ax, 2.8, 7.4, 2.4, 0.9, "DINOv2", C_FROZEN)
    box(ax, 2.8, 6.4, 2.4, 0.9, "SigLIP", C_FROZEN)
    arrow(ax, 2.2, 7.4, 2.8, 7.85)
    arrow(ax, 2.2, 6.9, 2.8, 6.85)

    # Concat & SE-bottleneck (perceptual stream)
    box(ax, 5.7, 6.8, 2.3, 1.4,
        "SE-bottleneck\n→ 256 perceptual\ntokens", C_MEMORY, fontsize=8.5)
    arrow(ax, 5.2, 7.85, 5.7, 7.7)
    arrow(ax, 5.2, 6.85, 5.7, 7.2)

    # LLaMA-7B (cognitive stream)
    box(ax, 5.7, 4.0, 2.3, 2.4,
        "LLaMA-7B\n\nFull 32 layers\n\nEOS @ layer 32\n→ 1 cognitive\ntoken", C_FROZEN, fontsize=8.5)
    arrow(ax, 5.2, 7.0, 5.7, 6.0, label="vision\ntokens")
    arrow(ax, 2.2, 5.25, 5.7, 5.25, label="instruction")

    # Banks
    box(ax, 9.0, 6.5, 1.8, 1.7,
        "Perceptual\nbank\n[L × 256 × dp]\n(consolidate)", C_MEMORY, fontsize=8)
    box(ax, 9.0, 4.4, 1.8, 1.7,
        "Cognitive\nbank\n[L × 1 × dc]\n(consolidate)", C_MEMORY, fontsize=8)

    arrow(ax, 8.0, 7.5, 9.0, 7.4, label="write")
    arrow(ax, 8.0, 5.2, 9.0, 5.25, label="write")

    # Two cross-attentions
    box(ax, 11.4, 6.5, 1.8, 1.7,
        "Cross-attn\n+ TE(t)\n→ H_per", C_MEMORY, fontsize=8.5)
    box(ax, 11.4, 4.4, 1.8, 1.7,
        "Cross-attn\n+ TE(t)\n→ H_cog", C_MEMORY, fontsize=8.5)
    arrow(ax, 10.8, 7.4, 11.4, 7.4, label="read")
    arrow(ax, 10.8, 5.25, 11.4, 5.25, label="read")

    # Action expert with two layer types
    box(ax, 13.6, 5.0, 1.3, 3.0,
        "Action\nexpert\n\nperc-attn\n+\ncog-attn\nlayers", C_TRAIN, fontsize=8.5)
    arrow(ax, 13.2, 7.4, 13.6, 6.7)
    arrow(ax, 13.2, 5.25, 13.6, 5.6)

    # Title + annotations
    ax.text(7.5, 8.7, "MemoryVLA — two-stream Perceptual-Cognitive Memory Bank",
            ha="center", fontsize=13, weight="bold")
    ax.text(7.5, 3.2,
            "Key facts: cognitive token = 1 token at LLaMA-7B EOS (full 32 layers).\n"
            "Perceptual stream bypasses LLM entirely (256 tokens from SE-bottleneck).\n"
            "Two banks → two cross-attentions → two distinct attention layers in action expert.",
            ha="center", fontsize=9, style="italic", color="#444")
    ax.text(7.5, 1.5,
            "Why this doesn't directly transfer to SmolVLA: SmolVLA truncates at 16/32 layers.\n"
            "There is no equivalent 'cognitive' representation available in our pipeline.",
            ha="center", fontsize=9, color=C_ACCENT)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_memoryvla.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig02_memoryvla.png")


# --------------------------------------------------------------------
# Figure 3 — V4 implementation
# --------------------------------------------------------------------
def fig_v4():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.axis("off")

    # Layer column
    layer_x = 1.0
    layer_w = 1.6
    layers_lower = [(layer_x, 7 - i*0.5, layer_w, 0.45,
                     f"Layer {i}", C_FROZEN, 8) for i in range(8)]
    for x, y, w, h, t, c, fs in layers_lower:
        box(ax, x, y, w, h, t, c, fontsize=fs)

    # Memory module (between layer 7 and layer 8)
    mem_x, mem_y, mem_w, mem_h = 4.5, 2.5, 6.5, 4.0
    bbox = FancyBboxPatch(
        (mem_x, mem_y), mem_w, mem_h,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        linewidth=2.0, edgecolor=C_ACCENT, facecolor=C_MEMORY, alpha=0.4,
    )
    ax.add_patch(bbox)
    ax.text(mem_x + mem_w / 2, mem_y + mem_h - 0.3,
            "MEMORY MODULE  (trainable)", ha="center", fontsize=11,
            weight="bold", color=C_ACCENT)

    box(ax, mem_x + 0.3, mem_y + 2.4, 2.5, 1.0,
        "Memory bank\n[16 entries × 170 tokens × 576]\nFIFO eviction\nDetached → CPU", C_NEUTRAL, fontsize=8)
    box(ax, mem_x + 3.3, mem_y + 2.4, 2.7, 1.0,
        "Cross-attention retrieval\nquery=current prefix\nkey=memory + temporal_PE\nvalue=memory", C_NEUTRAL, fontsize=8)
    box(ax, mem_x + 0.3, mem_y + 0.8, 2.5, 1.0,
        "memory_proj (Linear)\nzero-init at start", C_NEUTRAL, fontsize=8.5)
    box(ax, mem_x + 3.3, mem_y + 0.8, 2.7, 1.0,
        "Residual gate\nfused = current\n+ memory_proj(retrieved)", C_NEUTRAL, fontsize=8.5)

    arrow(ax, mem_x + 1.5, mem_y + 2.4, mem_x + 1.5, mem_y + 1.8, label="read")
    arrow(ax, mem_x + 4.6, mem_y + 2.4, mem_x + 4.6, mem_y + 1.8)
    arrow(ax, mem_x + 2.8, mem_y + 1.3, mem_x + 3.3, mem_y + 1.3)

    # Upper layers
    layer_x2 = 12.5
    layers_upper = [(layer_x2, 7 - i*0.5, layer_w, 0.45,
                     f"Layer {i+8}", C_FROZEN, 8) for i in range(8)]
    for x, y, w, h, t, c, fs in layers_upper:
        box(ax, x, y, w, h, t, c, fontsize=fs)

    # Highlight injection layer
    ax.add_patch(Rectangle((layer_x2, 6.55), layer_w, 0.45,
                           facecolor=C_ACCENT, alpha=0.3,
                           edgecolor=C_ACCENT, linewidth=2))
    ax.text(layer_x2 + layer_w / 2, 7.05, "Layer 8 (injection)",
            ha="center", fontsize=8, color=C_ACCENT, weight="bold")

    # Connectors
    arrow(ax, layer_x + layer_w, 3.5, mem_x, 4.5, label="hidden\nstates")
    arrow(ax, mem_x + mem_w, 4.5, layer_x2, 6.7, label="augmented\nprefix")

    # Action expert
    box(ax, 14.4, 1.5, 1.4, 3.0,
        "Action\nexpert\n\n(trainable)\n\nFlow\nmatching", C_TRAIN, fontsize=8.5)
    arrow(ax, 13.5, 3.0, 14.4, 3.0, label="KV cache")

    # Labels
    ax.text(layer_x + layer_w / 2, 7.7,
            "VLM lower\n(frozen)", ha="center", fontsize=9, weight="bold")
    ax.text(layer_x2 + layer_w / 2, 7.7,
            "VLM upper\n(frozen)", ha="center", fontsize=9, weight="bold")

    ax.text(8, 9.4, "V4 — current implementation",
            ha="center", fontsize=14, weight="bold")
    ax.text(8, 0.8,
            "Bank stores ENTIRE prefix per timestep (~170 tokens × 16 entries = 2,720 keys at retrieval).\n"
            "Trains in expert_finetune mode at effective batch=1 (B=1 forward, grad_accum_steps=1 default).\n"
            "Result: 73.25% overall vs 87.75% baseline; libero_10 hardest hit (44 vs 72).",
            ha="center", fontsize=9, color="#333")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_v4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig03_v4.png")


# --------------------------------------------------------------------
# Figure 4 — V4 results bar chart
# --------------------------------------------------------------------
def fig_results():
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "Overall"]
    baseline_v2 = [84, 99, 96, 72, 87.75]
    v4_bypass = [72, 96, 82, 54, 76.00]
    v4_memory = [74, 96, 79, 44, 73.25]

    x = np.arange(len(suites))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width, baseline_v2, width, label="V2 baseline (no memory)",
                color="#43a047", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x, v4_bypass, width, label="V4 bypass (memory off)",
                color="#fbc02d", edgecolor="black", linewidth=0.7)
    b3 = ax.bar(x + width, v4_memory, width, label="V4 memory ON",
                color="#e53935", edgecolor="black", linewidth=0.7)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.7, f"{h:.1f}",
                    ha="center", fontsize=8.5)

    ax.set_ylabel("Success rate (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(suites, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title("V4 results — per-suite success vs baseline\n"
                 "Two gaps: −11.75pp baseline regression (training mode) "
                 "+ −2.75pp memory cost",
                 fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate libero_10 hit
    ax.annotate("Worst hit:\nlong-horizon",
                xy=(3 + width, 44), xytext=(3.5, 25),
                fontsize=9, color=C_ACCENT, weight="bold",
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.5))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig04_results.png")


# --------------------------------------------------------------------
# Figure 5 — Bank key count comparison
# --------------------------------------------------------------------
def fig_bank_keys():
    options = ["V4\n(full prefix)",
               "V5 mean_pool\n(Option B)",
               "V5 compressor\nn=1",
               "V5 compressor\nn=4",
               "V5 compressor\nn=16",
               "V5 two-stream\n(Option D)"]
    keys = [2720, 16, 16, 64, 256, 272]
    colors = ["#e53935", "#1e88e5", "#42a5f5", "#1e88e5", "#1565c0", "#7b1fa2"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(options, keys, color=colors, edgecolor="black", linewidth=0.7)
    for bar, k in zip(bars, keys):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.15,
                f"{k:,}", ha="center", fontsize=10, weight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("Bank keys at retrieval (log scale)", fontsize=11)
    ax.set_ylim(8, 8000)
    ax.set_title("Bank key count: V4 vs V5 options\n"
                 "All assume bank_max_size=16; difference is tokens/entry",
                 fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.axhline(2720, color="#e53935", ls=":", alpha=0.5, lw=1)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_bank_keys.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig05_bank_keys.png")


# --------------------------------------------------------------------
# Figure 6 — Train vs inference bank fill rate
# --------------------------------------------------------------------
def fig_bank_fill():
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5))

    # Training timeline
    ax = axes[0]
    ax.set_title("V4 training (step_increment=1, write_stride=1)",
                 fontsize=11, weight="bold")
    ax.set_xlim(-2, 220); ax.set_ylim(-0.5, 18)
    ax.set_xlabel("frame")
    ax.set_ylabel("bank entry")

    # Show all writes (every frame)
    write_times = list(range(0, 200))
    bank_state = []
    for t in write_times:
        bank_state.append(t)
        if len(bank_state) > 16:
            bank_state.pop(0)
    # plot final state
    ax.scatter(write_times[:16], range(16),
               c="#1e88e5", s=20, label="bank entries (early episode)")
    ax.scatter(write_times[16:], [16] * (len(write_times) - 16),
               c="#90a4ae", s=8, alpha=0.3, label="evicted (FIFO)")
    ax.text(100, 17, "Time deltas at retrieval, end of episode: {1, 2, ..., 16}",
            fontsize=10, color="#1e88e5", weight="bold",
            ha="center")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Inference timeline
    ax = axes[1]
    ax.set_title("V4 inference (step_increment=10, callback fires every chunk)",
                 fontsize=11, weight="bold")
    ax.set_xlim(-2, 220); ax.set_ylim(-0.5, 18)
    ax.set_xlabel("env step")
    ax.set_ylabel("bank entry")

    # Show writes only every 10 env steps
    write_times = list(range(0, 200, 10))
    ax.scatter(write_times, range(len(write_times)),
               c=C_ACCENT, s=40, label="bank writes (sparse)")
    ax.text(100, 17,
            "Time deltas at retrieval: {10, 20, 30, ..., 160}  ← OOD: never seen at training",
            fontsize=10, color=C_ACCENT, weight="bold", ha="center")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("Drawback 2: train/inference temporal-PE distribution shift",
                 fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_bank_fill.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig06_bank_fill.png")


# --------------------------------------------------------------------
# Figure 7 — Gradient diversity comparison
# --------------------------------------------------------------------
def fig_gradient_diversity():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # V2 baseline
    ax = axes[0]
    ax.set_title("V2 baseline — random batched (B=32)",
                 fontsize=11, weight="bold", color="#43a047")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    rng = np.random.RandomState(42)
    # Show 32 dots from different episodes, scattered
    eps_x = rng.uniform(0.5, 9.5, 32)
    eps_y = rng.uniform(0.5, 9.5, 32)
    eps_colors = rng.choice([
        "#ef5350", "#42a5f5", "#66bb6a", "#ab47bc", "#ffa726",
        "#26a69a", "#ec407a", "#7e57c2"], 32)
    ax.scatter(eps_x, eps_y, c=eps_colors, s=120, alpha=0.7, edgecolor="black")
    ax.text(5, 9.5, "32 random (episode, frame) per gradient step",
            ha="center", fontsize=10, weight="bold")
    ax.text(5, 0.2, "8 distinct episode colors → high scene diversity",
            ha="center", fontsize=9, style="italic", color="#43a047")

    # V4
    ax = axes[1]
    ax.set_title("V4 — episode-sequential (B=1, grad_accum=1)",
                 fontsize=11, weight="bold", color=C_ACCENT)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    # Show 1 dot
    ax.scatter([5], [5], c="#ef5350", s=400, alpha=0.7, edgecolor="black")
    ax.text(5, 9.5, "1 frame from 1 episode per gradient step",
            ha="center", fontsize=10, weight="bold")
    ax.text(5, 0.2, "Single-color → near-zero scene diversity per step",
            ha="center", fontsize=9, style="italic", color=C_ACCENT)
    ax.text(5, 3.5, "32× collapse vs baseline",
            ha="center", fontsize=12, weight="bold", color=C_ACCENT)

    plt.suptitle(
        "Drawback 3: gradient-diversity collapse  "
        "(likely dominant cause of V4's −12pp regression)",
        fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_gradient_diversity.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("fig07_gradient_diversity.png")


# --------------------------------------------------------------------
# Figure 8 — V5 window loader vs V4 loader
# --------------------------------------------------------------------
def fig_window_loader():
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.5))

    # V4
    ax = axes[0]
    ax.set_title("V4 loader — full episode visit, then boundary",
                 fontsize=11, weight="bold", color=C_ACCENT)
    ax.set_xlim(-2, 220); ax.set_ylim(0, 4)
    ax.axis("off")
    # Episode A: 0-100
    ax.add_patch(Rectangle((0, 1.5), 100, 1.5,
                           facecolor="#ef9a9a", edgecolor="black"))
    ax.text(50, 2.25, "Episode A — all 100 frames", ha="center",
            fontsize=10, weight="bold")
    # Boundary
    ax.axvline(100, color="black", linestyle="--", lw=1.5)
    ax.text(100, 0.5, "boundary", ha="center", fontsize=8)
    # Episode B: 100-200
    ax.add_patch(Rectangle((100, 1.5), 100, 1.5,
                           facecolor="#90caf9", edgecolor="black"))
    ax.text(150, 2.25, "Episode B — all 100 frames", ha="center",
            fontsize=10, weight="bold")
    # Annotate accumulation
    for i in range(0, 200, 32):
        ax.add_patch(Rectangle((i, 3.2), 32, 0.4,
                               facecolor="#fff3e0", edgecolor="#ef6c00"))
        ax.text(i + 16, 3.4, "1 grad step", ha="center", fontsize=7)
    ax.text(110, 3.85, "→ within-episode autocorrelation",
            fontsize=9, color=C_ACCENT)

    # V5 window
    ax = axes[1]
    ax.set_title("V5 window loader — N frames per episode visit, random offset",
                 fontsize=11, weight="bold", color="#43a047")
    ax.set_xlim(-2, 220); ax.set_ylim(0, 4)
    ax.axis("off")
    # Many short windows from random episodes
    colors = ["#ef9a9a", "#90caf9", "#a5d6a7", "#ce93d8", "#ffcc80", "#80cbc4"]
    for i, x in enumerate(range(0, 200, 32)):
        ax.add_patch(Rectangle((x, 1.5), 32, 1.5,
                               facecolor=colors[i % len(colors)], edgecolor="black"))
        ax.text(x + 16, 2.25, f"Ep {chr(65+i)}", ha="center",
                fontsize=10, weight="bold")
        ax.add_patch(Rectangle((x, 3.2), 32, 0.4,
                               facecolor="#fff3e0", edgecolor="#ef6c00"))
        ax.text(x + 16, 3.4, "1 grad step", ha="center", fontsize=7)
    ax.text(110, 3.85, "→ different episode per gradient step",
            fontsize=9, color="#43a047", weight="bold")

    plt.suptitle("V5 fix #1: window loader (Option 2)",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig08_window_loader.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("fig08_window_loader.png")


# --------------------------------------------------------------------
# Figure 9 — V5 fixes overview
# --------------------------------------------------------------------
def fig_v5_overview():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(8, 9.4, "V5 fixes — five orthogonal changes",
            ha="center", fontsize=14, weight="bold")

    fixes = [
        ("Window Loader", "Drawback 3", "max_window_size=32\n+ grad_accum=32",
         "Cross-episode\ndiversity per\noptimizer step", "#fff3e0"),
        ("Write Stride", "Drawback 2", "write_stride=50",
         "Bank fill rate\nmatches inference\n→ no PE OOD", "#e8f5e9"),
        ("Mean Pool\n(Option B)", "Drawback 1", "compression_mode\n=mean_pool",
         "1 token/entry\n→ 16 keys total\nzero params", "#e3f2fd"),
        ("Perceiver\n(Option A)", "Drawback 1", "use_compressor=True\nn_slots=4",
         "Learned compression\n→ 64 keys total\n~50K params", "#f3e5f5"),
        ("Two-stream\n(Option D)", "Drawback 1+4", "two_stream=True\np=16, t=1\nn_image=132",
         "Explicit perc/task\nbudget → 272 keys\n~100K params", "#fce4ec"),
    ]

    for i, (name, target, change, effect, color) in enumerate(fixes):
        x = 0.3 + i * 3.15
        # Header
        box(ax, x, 7.5, 2.9, 1.2,
            f"{name}\n\nfixes {target}", color, fontsize=10)
        # Change
        box(ax, x, 5.0, 2.9, 2.2,
            f"What it does:\n\n{change}", "#fafafa", fontsize=9)
        # Effect
        box(ax, x, 2.4, 2.9, 2.3,
            f"Effect:\n\n{effect}", "#fafafa", fontsize=9)

    ax.text(8, 1.3,
            "All five fixes live on a single merged branch (claude/feature/v5-all-fixes).\n"
            "Each is independently togglable via config flags. Default behavior = bit-identical to V4.",
            ha="center", fontsize=9.5, style="italic")
    ax.text(8, 0.5,
            "Run plan: Run 0 (window only) → Run 1 (window+stride+mean_pool) → "
            "Run 2A or 2B if Run 1 plateaus.",
            ha="center", fontsize=9.5, color="#444")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig09_v5_overview.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("fig09_v5_overview.png")


# --------------------------------------------------------------------
# Figure 10 — Run plan flowchart
# --------------------------------------------------------------------
def fig_run_plan():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(7, 9.4, "V5 run plan & decision rules", ha="center",
            fontsize=14, weight="bold")

    box(ax, 5.5, 7.5, 3.0, 1.0,
        "Run 0 — diagnostic\n\nbatch fix only (window + grad_accum)",
        "#fff3e0", fontsize=10)
    box(ax, 0.3, 5.0, 3.5, 1.5,
        "If Run 0 ≥ 80% overall:\n\nbatch fix alone closed gap.\n"
        "Run 1 to test additional gain.", "#e8f5e9", fontsize=9)
    box(ax, 5.5, 5.0, 3.0, 1.5,
        "Run 1 — kitchen sink\n\nRun 0 + write_stride + mean_pool\n\n"
        "(all cheap fixes)", "#fff3e0", fontsize=9.5)
    box(ax, 10.2, 5.0, 3.5, 1.5,
        "If Run 0 < 76% overall:\n\nbatch fix insufficient.\n"
        "Run 1 must do the architectural work.", "#ffebee", fontsize=9)

    arrow(ax, 7, 7.5, 5.5, 6.5)
    arrow(ax, 7, 7.5, 7, 6.5)
    arrow(ax, 7, 7.5, 9.5, 6.5)

    box(ax, 0.3, 2.5, 3.5, 1.5,
        "Run 2A — Perceiver\n\nReplace mean_pool with\nlearned compressor (n=4)",
        "#f3e5f5", fontsize=9.5)
    box(ax, 5.5, 2.5, 3.0, 1.5,
        "Run 2B — Two-stream\n\nReplace mean_pool with\n"
        "perceptual + task-anchor split", "#fce4ec", fontsize=9.5)
    box(ax, 10.2, 2.5, 3.5, 1.5,
        "Stop with Run 1\n\nIf Run 1 ≥ baseline,\nfull eval and ship.",
        "#e8f5e9", fontsize=9.5)

    arrow(ax, 7, 5.0, 2, 4.0, label="if Run 1\nplateaus")
    arrow(ax, 7, 5.0, 7, 4.0, label="if Run 1\nplateaus on\nlibero_10")
    arrow(ax, 7, 5.0, 12, 4.0, label="if Run 1\nbeats baseline")

    box(ax, 5.5, 0.4, 3.0, 1.0,
        "Final: full eval (400 ep)\non best config",
        "#c8e6c9", fontsize=10)
    arrow(ax, 2, 2.5, 6.5, 1.4)
    arrow(ax, 7, 2.5, 7, 1.4)
    arrow(ax, 12, 2.5, 7.5, 1.4)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_run_plan.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig10_run_plan.png")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    fig_smolvla()
    fig_memoryvla()
    fig_v4()
    fig_results()
    fig_bank_keys()
    fig_bank_fill()
    fig_gradient_diversity()
    fig_window_loader()
    fig_v5_overview()
    fig_run_plan()
    print("\nAll figures generated in", FIG_DIR)
