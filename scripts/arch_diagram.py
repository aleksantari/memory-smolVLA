#!/usr/bin/env python
"""Architecture comparison diagram: memory-augmented SmolVLA V6 / V7 / V8.

Renders docs/arch_v6_v7_v8.png. Pure matplotlib, CPU only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ----------------------------------------------------------------- palette --
INK      = "#1a1f2b"
INK2     = "#4a5568"
SURF     = "#ffffff"
PANEL    = "#f5f6f8"
PANEL_E  = "#d4d8de"
FROZEN   = "#e2e5ea"
FROZEN_E = "#9aa3b0"
TRAIN    = "#fdf3e0"
TRAIN_E  = "#c98f1b"
NEUT     = "#eef0f3"
NEUT_E   = "#8b93a1"
MEM      = "#e8eef9"
MEM_E    = "#3a6ea8"
# Okabe-Ito-derived, colorblind-safe version hues
V6C, V6D = "#cfe4f2", "#0072B2"
V7C, V7D = "#fcecd1", "#D55E00"
V8C, V8D = "#d8efe6", "#009E73"

FS_TITLE, FS_CAP, FS_PANEL, FS_HEAD, FS_BODY, FS_SMALL = 26, 14.5, 17, 14, 11.5, 10

# ---------------------------------------------------------------- canvas ----
W, H = 226, 150
fig = plt.figure(figsize=(22.6, 15.0), dpi=160)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect("equal"); ax.axis("off")
fig.patch.set_facecolor(SURF)

# ---------------------------------------------------------------- helpers ---
def box(x, y, w, h, fc, ec, lw=1.4, r=1.2, hatch=None, ls="-", z=3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, hatch=hatch, ls=ls, zorder=z)
    if hatch:
        p.set_hatch_linewidth(0.6)
    ax.add_patch(p)
    return p

def txt(x, y, s, size=FS_BODY, color=INK, ha="center", va="center",
        weight="normal", z=6, style="normal", ls=1.25, bg=None):
    kw = {}
    if bg is not None:
        kw["bbox"] = dict(boxstyle="round,pad=0.25", fc=bg, ec="none")
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   weight=weight, zorder=z, style=style, linespacing=ls, **kw)

def arrow(p0, p1, color=INK2, lw=1.8, z=4, style="-", shrinkA=1.5, shrinkB=1.5,
          rad=0.0, mut=11):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mut,
                        color=color, lw=lw, zorder=z, linestyle=style,
                        shrinkA=shrinkA, shrinkB=shrinkB,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    return a

def tag(x, y, s, fc, ec, size=FS_SMALL, tc=None, weight="bold", pad=0.35):
    return ax.text(x, y, s, fontsize=size, color=tc or ec, ha="center",
                   va="center", weight=weight, zorder=7,
                   bbox=dict(boxstyle=f"round,pad={pad}", fc=fc, ec=ec, lw=1.1))

def token_row(cx, cy, n, size, fc, ec, gap_=0.5, z=6):
    tot = n*size + (n-1)*gap_
    x = cx - tot/2
    for i in range(n):
        ax.add_patch(Rectangle((x+i*(size+gap_), cy), size, size,
                     fc=fc, ec=ec, lw=0.8, zorder=z))
    return tot

# ----------------------------------------------------------------- title ----
txt(W/2, H-4.2, "Memory-Augmented SmolVLA — V6 → V7 → V8",
    size=FS_TITLE, weight="bold")
txt(W/2, H-8.8,
    "One shared backbone (frozen SmolVLM2 + action expert + FullSeqMemBank injected at VLM layer 15); "
    "the versions differ only in what each frame writes into the memory bank.",
    size=FS_CAP, color=INK2)

# ------------------------------------------------------------------ legend --
items = [("Frozen (SmolVLM2)", FROZEN, FROZEN_E, "///"),
         ("Trained from scratch", TRAIN, TRAIN_E, None),
         ("Memory path", MEM, MEM_E, None)]
widths = [4.2 + len(lab)*1.02 + 6 for lab, *_ in items]
lgx = W/2 - sum(widths)/2
lg_y = H-12.8
for (lab, fc, ec, ht), wd in zip(items, widths):
    box(lgx, lg_y-1.1, 3.2, 2.2, fc, ec, r=0.4, hatch=ht, z=6)
    txt(lgx+4.2, lg_y, lab, ha="left", size=FS_SMALL, color=INK2)
    lgx += wd

# ============================================================ TOP PANEL =====
TP_Y0, TP_Y1 = 93.5, 135
box(3, TP_Y0, W-6, TP_Y1-TP_Y0, PANEL, PANEL_E, lw=1.2, r=1.6, z=1)
txt(7, TP_Y1-3.2, "SHARED BACKBONE  (identical in V6 / V7 / V8)",
    size=FS_PANEL, weight="bold", ha="left")

# ---- inputs
in_x, in_w, in_h = 7, 26, 5.2
labels = ["RGB cam — agentview", "RGB cam — wrist",
          "Language instruction", "Robot state (8-dim)"]
in_ys = [TP_Y1-12.5, TP_Y1-19.0, TP_Y1-25.5, TP_Y1-32.0]
for lab, iy in zip(labels, in_ys):
    box(in_x, iy, in_w, in_h, NEUT, NEUT_E)
    txt(in_x+in_w/2, iy+in_h/2, lab, size=FS_SMALL)

# ---- embed box
emb_x, emb_w = 40, 20
emb_y, emb_h = TP_Y1-27.5, 12
box(emb_x, emb_y, emb_w, emb_h, NEUT, NEUT_E)
txt(emb_x+emb_w/2, emb_y+emb_h/2,
    "Tokenize\n& embed\n→ prefix tokens\n(dim 960)", size=FS_SMALL)
for iy, fy in zip(in_ys, [emb_y+emb_h-2.2, emb_y+emb_h-4.4,
                          emb_y+4.4, emb_y+2.2]):
    arrow((in_x+in_w, iy+in_h/2), (emb_x, fy))

# ---- VLM stack
vlm_x, vlm_w = 68, 34
vlm_y, vlm_h = TP_Y0+3.5, 31
box(vlm_x, vlm_y, vlm_w, vlm_h, FROZEN, FROZEN_E, hatch="///", lw=1.6)
txt(vlm_x+vlm_w/2, vlm_y+vlm_h-2.4, "SmolVLM2  (VLM)", weight="bold", size=FS_HEAD)
tag(vlm_x+vlm_w/2, vlm_y+vlm_h, "FROZEN", FROZEN, FROZEN_E, tc=INK2)
b_x, b_w = vlm_x+2.5, vlm_w-5
bands = [("layers 1–14", 6.0, NEUT, NEUT_E, None),
         ("MEMORY INJECTION\n(before layer-15 input-layernorm)", 5.4, MEM, MEM_E, "inj"),
         ("layer 15", 3.6, NEUT, NEUT_E, None),
         ("layer 16", 3.6, NEUT, NEUT_E, None)]
cy = vlm_y+vlm_h-5.2
inj_c = None
for lab, bh, fc, ec, kind in bands:
    cy -= bh + 1.0
    box(b_x, cy, b_w, bh, fc, ec, r=0.7)
    if kind:
        txt(b_x+b_w/2, cy+bh/2+0.9, "MEMORY INJECTION", size=FS_SMALL,
            color=MEM_E, weight="bold")
        txt(b_x+b_w/2, cy+bh/2-1.2, "(before layer-15 input-layernorm)",
            size=FS_SMALL-1.5, color=MEM_E)
        inj_c = (b_x+b_w, cy+bh/2)
    else:
        txt(b_x+b_w/2, cy+bh/2, lab, size=FS_SMALL)
arrow((emb_x+emb_w, emb_y+emb_h/2), (vlm_x, vlm_y+vlm_h*0.62))

# ---- memory bank callout
mb_x, mb_w = 114, 44
mb_y, mb_h = TP_Y0+2.5, 29
box(mb_x, mb_y, mb_w, mb_h, MEM, MEM_E, lw=1.8)
txt(mb_x+mb_w/2, mb_y+mb_h-2.9, "FullSeqMemBank", weight="bold",
    size=FS_HEAD, color=MEM_E)
tag(mb_x+11, mb_y+mb_h, "TRAINED FROM SCRATCH", TRAIN, TRAIN_E,
    size=FS_SMALL-1.5)
sub = [("RETRIEVE — prefix queries cross-attend bank K/V;\nsinusoidal timestep PE on keys (2 stacked blocks)", 6.2),
       ("FUSE — learned per-token sigmoid gate:\nfused = g · current + (1−g) · retrieved", 6.2),
       ("CONSOLIDATE — ToMe merges most-similar adjacent\npair when bank > mem_length · writes detached", 6.2)]
sy = mb_y+mb_h-5.0
for lab, sh in sub:
    sy -= sh + 0.9
    box(mb_x+2, sy, mb_w-4, sh, "#f7fafd", MEM_E, r=0.7, lw=1.0)
    txt(mb_x+2+(mb_w-4)/2, sy+sh/2, lab, size=FS_SMALL-0.8)
txt(mb_x+mb_w/2, mb_y+1.6, "per-episode bank · capacity = mem_length entries",
    size=FS_SMALL-0.8, color=INK2, style="italic")
# arrows injection <-> bank, labels in the gap with white bg
arrow(inj_c, (mb_x, mb_y+mb_h*0.66), color=MEM_E, lw=2.0, rad=0.10)
arrow((mb_x, mb_y+mb_h*0.34), inj_c, color=MEM_E, lw=2.0, rad=0.10)
mid_gap = (inj_c[0]+mb_x)/2
txt(mid_gap, mb_y+mb_h*0.80, "query + write", size=FS_SMALL-1, color=MEM_E,
    bg=PANEL)
txt(mid_gap, mb_y+mb_h*0.20, "fused back", size=FS_SMALL-1, color=MEM_E,
    bg=PANEL)

# ---- action expert
ae_x, ae_w = 166, 28
ae_y, ae_h = TP_Y0+11, 23.5
box(ae_x, ae_y, ae_w, ae_h, TRAIN, TRAIN_E, lw=1.6)
txt(ae_x+ae_w/2, ae_y+ae_h-3.0, "Action Expert", weight="bold", size=FS_HEAD)
tag(ae_x+ae_w/2, ae_y+ae_h, "TRAINED FROM SCRATCH", TRAIN, TRAIN_E,
    size=FS_SMALL-1.5)
txt(ae_x+ae_w/2, ae_y+ae_h/2-1.6,
    "width-½ copy of the VLM,\ninterleaved per layer;\ncross-attends the VLM's\nper-layer K/V", size=FS_SMALL)
# K/V arrow: VLM top-right -> over the bank -> expert left edge (near top)
arrow((vlm_x+vlm_w-4, vlm_y+vlm_h), (ae_x, ae_y+ae_h-2.0),
      color=INK2, lw=2.0, rad=-0.06, shrinkA=0.5)
txt(132, 133.3, "per-layer K/V", size=FS_SMALL, color=INK2, bg=PANEL)

# ---- flow head + chunk
fh_x, fh_w = 200, 21
fh_y, fh_h = TP_Y0+21, 8
box(fh_x, fh_y, fh_w, fh_h, TRAIN, TRAIN_E)
txt(fh_x+fh_w/2, fh_y+fh_h/2, "Flow-matching\nhead", size=FS_SMALL)
ac_y, ac_h = TP_Y0+7, 8
box(fh_x, ac_y, fh_w, ac_h, NEUT, NEUT_E)
txt(fh_x+fh_w/2, ac_y+ac_h/2, "Action chunk\npredict 50 · execute 10",
    size=FS_SMALL)
arrow((ae_x+ae_w, fh_y+fh_h/2), (fh_x, fh_y+fh_h/2))
arrow((fh_x+fh_w/2, fh_y), (fh_x+fh_w/2, ac_y+ac_h))

# ========================================================== MIDDLE PANEL ====
MP_Y0, MP_Y1 = 39.5, 89.5
box(3, MP_Y0, W-6, MP_Y1-MP_Y0, PANEL, PANEL_E, lw=1.2, r=1.6, z=1)
txt(7, MP_Y1-3.2, "WHAT LIVES IN THE MEMORY BANK  (the only thing that changes)",
    size=FS_PANEL, weight="bold", ha="left")

col_w, gap = 70, 3
c_xs = [5.5, 5.5+col_w+gap, 5.5+2*(col_w+gap)]
col_y0, col_y1 = MP_Y0+2.5, MP_Y1-6.5

def col_frame(cx0, name, fill, dark):
    box(cx0, col_y0, col_w, col_y1-col_y0, SURF, dark, lw=1.8, r=1.2, z=2)
    box(cx0+1.5, col_y1-6.3, col_w-3, 4.8, fill, dark, r=0.8, z=3)
    txt(cx0+col_w/2, col_y1-3.9, name, weight="bold", size=FS_HEAD-1,
        color=dark)

def cfg_lines(cx0, ty, lines):
    for line, emph in lines:
        txt(cx0+3.5, ty, line, size=FS_SMALL, ha="left",
            color=INK, weight="bold" if emph else "normal")
        ty -= 2.7

# ---------------- V6 column ----------------
cx0 = c_xs[0]
col_frame(cx0, "V6 — diversity baseline", V6C, V6D)
txt(cx0+col_w/2, col_y1-9.2, "stores the FULL prefix sequence per frame",
    weight="bold", size=FS_BODY, color=V6D)
# write path: prefix box -> stored grid
pb_y = col_y1-19.5
box(cx0+4, pb_y, 17, 6.5, NEUT, NEUT_E, r=0.8)
txt(cx0+4+8.5, pb_y+3.25, "current\nprefix", size=FS_SMALL)
arrow((cx0+21.5, pb_y+3.25), (cx0+27.5, pb_y+3.25))
txt(cx0+24.5, pb_y+5.2, "as-is", size=FS_SMALL-1, color=INK2, style="italic")
for r_ in range(3):
    token_row(cx0+47.5, pb_y+0.6+r_*2.0, 24, 1.25, V6C, V6D, gap_=0.3)
txt(cx0+47.5, pb_y-1.9, "~hundreds of tokens / entry  (heavy)",
    size=FS_SMALL-1, color=INK2, style="italic")
cfg_lines(cx0, pb_y-7.5, [
    ("mem_length = 4     group_size = 4  (== mem_length)", False),
    ("→ ToMe NEVER fires in training, but ~48 merges at eval", False),
    ("→ TRAIN / EVAL CONSOLIDATION MISMATCH", True),
    ("num_groups = 12  (episode diversity)", False)])

# ---------------- V7 column ----------------
cx0 = c_xs[1]
col_frame(cx0, "V7 = V6 + compression + consolidation-matching", V7C, V7D)
txt(cx0+col_w/2, col_y1-9.2, "stores 1 MEAN-POOLED token per frame",
    weight="bold", size=FS_BODY, color=V7D)
pb_y = col_y1-19.5
box(cx0+4, pb_y, 17, 6.5, NEUT, NEUT_E, r=0.8)
txt(cx0+4+8.5, pb_y+3.25, "current\nprefix", size=FS_SMALL)
mp_x = cx0+28
box(mp_x, pb_y+0.25, 15, 6, NEUT, NEUT_E, r=0.8)
txt(mp_x+7.5, pb_y+3.25, "mean-pool", size=FS_SMALL)
arrow((cx0+21.5, pb_y+3.25), (mp_x, pb_y+3.25))
arrow((mp_x+15, pb_y+3.25), (cx0+51, pb_y+3.25))
token_row(cx0+53.5, pb_y+2.0, 1, 2.6, V7C, V7D)
txt(cx0+53.5, pb_y-1.9, "1 token / entry  (light)",
    size=FS_SMALL-1, color=INK2, style="italic")
cfg_lines(cx0, pb_y-7.5, [
    ("mem_length = 4     group_size = 8  (> mem_length)", False),
    ("→ ToMe FIRES in training → mismatch FIXED", True),
    ("num_groups = 16  (more episode diversity)", False),
    ("memory itself adds +5 pp on libero_10", False)])

# ---------------- V8 column ----------------
cx0 = c_xs[2]
col_frame(cx0, "V8 = V7 + learned reasoning tokens", V8C, V8D)
txt(cx0+col_w/2, col_y1-9.2, "stores 8 LEARNED REASONING tokens per frame",
    weight="bold", size=FS_BODY, color=V8D)
pb_y = col_y1-19.5
box(cx0+3, pb_y, 14.5, 6.5, NEUT, NEUT_E, r=0.8)
txt(cx0+3+7.25, pb_y+3.25, "current\nprefix", size=FS_SMALL)
rs_x = cx0+22
box(rs_x, pb_y-0.5, 22, 7.5, TRAIN, TRAIN_E, r=0.8)
txt(rs_x+11, pb_y+3.25, "ReasoningSummaryHead\nPerceiver-Resampler\n8 latent queries",
    size=FS_SMALL-1)
arrow((cx0+17.5, pb_y+3.25), (rs_x, pb_y+3.25))
arrow((rs_x+22, pb_y+3.25), (cx0+48.5, pb_y+3.25))
tok_cx = cx0+58
token_row(tok_cx, pb_y+2.2, 8, 2.1, V8C, V8D, gap_=0.45)
txt(tok_cx, pb_y-1.9, "8 tokens / entry", size=FS_SMALL-1, color=INK2,
    style="italic")
# aux branch: dashed box fed by the reasoning tokens
fp_y = pb_y-11.5
box(cx0+14, fp_y, 52, 7, TRAIN, TRAIN_E, r=0.8, ls="--")
txt(cx0+14+26, fp_y+3.5,
    "NEW aux branch — FutureStatePredictor reads the reasoning\n"
    "tokens → predicts robot state a few steps ahead (PTP loss)",
    size=FS_SMALL-0.5)
arrow((tok_cx+7, pb_y+2.2), (tok_cx+7, fp_y+7), color=TRAIN_E, lw=1.8)
cfg_lines(cx0, fp_y-3.0, [
    ("L = flow_matching + 0.1 · aux — this aux loss is what trains", False),
    ("the summary head (bank writes are detached)", False),
    ("everything else = V7", True)])

# ========================================================== RESULTS STRIP ===
RS_Y0, RS_Y1 = 4, 35.5
box(3, RS_Y0, W-6, RS_Y1-RS_Y0, PANEL, PANEL_E, lw=1.2, r=1.6, z=1)
txt(7, RS_Y1-3.2, "RESULTS  (LIBERO success rate)", size=FS_PANEL,
    weight="bold", ha="left")

res = [("V6", V6C, V6D, 81.75, 52.0, "long-horizon libero_10 is the weak spot"),
       ("V7", V7C, V7D, 86.75, 74.0, "+22 pp on libero_10 vs V6  —  best config"),
       ("V8", V8C, V8D, 85.0, 64.0, "reasoning tokens hurt libero_10 (64 vs V7 74)")]
tile_y0, tile_y1 = RS_Y0+3.0, RS_Y1-6.5
for cx0, (name, fill, dark, ov, l10, note) in zip(c_xs, res):
    box(cx0, tile_y0, col_w, tile_y1-tile_y0, SURF, dark, lw=1.8, r=1.2, z=2)
    txt(cx0+4, (tile_y0+tile_y1)/2, name, weight="bold", size=FS_TITLE-4,
        color=dark, ha="left")
    bar_x0, bar_wmax = cx0+23, col_w-34
    if ov is not None:
        for (lab, val, yy) in [("Overall", ov, tile_y1-7.0),
                               ("libero_10", l10, tile_y1-12.0)]:
            txt(bar_x0-1.2, yy+1.0, lab, ha="right", size=FS_SMALL, color=INK2)
            ax.add_patch(Rectangle((bar_x0, yy), bar_wmax, 2.0, fc="#e8eaee",
                                   ec="none", zorder=4))
            ax.add_patch(Rectangle((bar_x0, yy), bar_wmax*val/100, 2.0,
                                   fc=dark, ec="none", zorder=5))
            txt(bar_x0+bar_wmax*val/100+1.2, yy+1.0, f"{val:g}%",
                ha="left", weight="bold", size=FS_SMALL, color=INK)
        if note:
            txt(bar_x0+bar_wmax/2, tile_y0+3.0, note, size=FS_SMALL,
                color=dark, style="italic")
    else:
        txt(bar_x0+bar_wmax/2, (tile_y0+tile_y1)/2+1.4, "training…",
            size=FS_HEAD, color=dark, weight="bold")
        txt(bar_x0+bar_wmax/2, (tile_y0+tile_y1)/2-2.4,
            "V7 recipe + reasoning tokens + PTP aux loss",
            size=FS_SMALL, color=INK2, style="italic")

fig.savefig("/home/tarmus/memory-smolVLA/docs/arch_v6_v7_v8.png",
            dpi=160, facecolor=SURF)
print("saved")
