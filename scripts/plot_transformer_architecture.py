#!/usr/bin/env python
"""
Generate publication-quality transformer architecture diagram.

This creates a block diagram of the StageBridge transformer architecture
showing the receiver-centered design with L-R integration.

Usage:
    python scripts/plot_transformer_architecture.py --output docs/figures/transformer_arch.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


# Publication color palette (colorblind-safe)
COLORS = {
    'recv': '#E69F00',      # Orange - receiver
    'sender': '#009E73',    # Teal - senders
    'hlca': '#56B4E9',      # Sky blue - HLCA
    'luca': '#D55E00',      # Vermillion - LuCA
    'lr': '#CC79A7',        # Pink - L-R tokens
    'relay': '#7570B3',     # Purple - relay
    'attn': '#F0E442',      # Yellow - attention
    'output': '#0072B2',    # Blue - output
    'bg': '#FAFAFC',        # Light background
    'text': '#333333',      # Dark text
}


def draw_token(ax, x, y, label, color, width=0.8, height=0.5, fontsize=7):
    """Draw a token box."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.7
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='black')


def draw_block(ax, x, y, label, color, width=2.5, height=0.9, fontsize=8, sublabel=None):
    """Draw an encoder block."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.03,rounding_size=0.15",
        facecolor=color, edgecolor='black', linewidth=1.2, alpha=0.3
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.12, label, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='black')
        ax.text(x, y - 0.15, sublabel, ha='center', va='center', fontsize=6,
                color='gray')
    else:
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='black')


def draw_arrow(ax, start, end, color='gray', style='->', linewidth=1.0):
    """Draw an arrow between points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=linewidth))


def create_architecture_diagram(output_path=None):
    """Create the transformer architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-3, 14)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Title
    ax.text(5, 13.5, 'StageBridge Transformer Architecture',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(5, 13.0, 'Receiver-Centered Communication Encoder with L-R Integration',
            ha='center', va='center', fontsize=9, color='gray')

    # ==========================================
    # SECTION 1: Input Tokens
    # ==========================================
    ax.text(0, 12.3, 'Input Tokens', fontsize=10, fontweight='bold', color='gray')

    # Background for tokens
    bg_box = FancyBboxPatch(
        (-0.5, 11.1), 10.5, 0.9,
        boxstyle="round,rounding_size=0.1",
        facecolor='#F5F5F8', edgecolor='none'
    )
    ax.add_patch(bg_box)

    # Token sequence
    draw_token(ax, 0.2, 11.5, 'Recv', COLORS['recv'])
    draw_token(ax, 1.3, 11.5, 'S₁', COLORS['sender'])
    draw_token(ax, 2.2, 11.5, 'S₂', COLORS['sender'], width=0.7)
    draw_token(ax, 3.0, 11.5, '...', COLORS['sender'], width=0.6)
    draw_token(ax, 3.8, 11.5, 'Sₖ', COLORS['sender'], width=0.7)

    draw_token(ax, 5.2, 11.5, 'LR₁', COLORS['lr'])
    draw_token(ax, 6.1, 11.5, 'LR₂', COLORS['lr'], width=0.7)
    draw_token(ax, 6.9, 11.5, '...', COLORS['lr'], width=0.6)

    draw_token(ax, 8.0, 11.5, 'Rly₁', COLORS['relay'])
    draw_token(ax, 8.9, 11.5, 'Rly₂', COLORS['relay'], width=0.7)
    draw_token(ax, 9.7, 11.5, '...', COLORS['relay'], width=0.6)

    # Labels below tokens
    ax.text(0.2, 10.9, 'Receiver', ha='center', fontsize=6, color='gray')
    ax.text(2.5, 10.9, 'Senders (k)', ha='center', fontsize=6, color='gray')
    ax.text(6.1, 10.9, 'L-R Pairs (n)', ha='center', fontsize=6, color='gray')
    ax.text(8.9, 10.9, 'Relay (m)', ha='center', fontsize=6, color='gray')

    # ==========================================
    # SECTION 2: Projections
    # ==========================================
    ax.text(0, 10.3, 'Projections', fontsize=10, fontweight='bold', color='gray')

    # Projection boxes
    draw_block(ax, 0.2, 9.6, 'Wᵣ', COLORS['recv'], width=0.8, height=0.5, fontsize=7)
    draw_block(ax, 2.5, 9.6, 'Wₛ + Type + Ring + Offset', COLORS['sender'],
               width=3.0, height=0.5, fontsize=6)
    draw_block(ax, 6.0, 9.6, 'W_lr + Type', COLORS['lr'], width=1.8, height=0.5, fontsize=6)
    draw_block(ax, 8.8, 9.6, 'W_rl + Type', COLORS['relay'], width=1.8, height=0.5, fontsize=6)

    # Arrows from tokens to projections
    draw_arrow(ax, (0.2, 11.1), (0.2, 9.9), COLORS['recv'])
    draw_arrow(ax, (2.5, 11.1), (2.5, 9.9), COLORS['sender'])
    draw_arrow(ax, (6.0, 11.1), (6.0, 9.9), COLORS['lr'])
    draw_arrow(ax, (8.8, 11.1), (8.8, 9.9), COLORS['relay'])

    # ==========================================
    # SECTION 3: Sender Encoder
    # ==========================================
    ax.text(0, 8.6, 'Sender Encoder', fontsize=10, fontweight='bold', color='gray')

    draw_block(ax, 2.5, 7.8, 'Self-Attention (SAB)', COLORS['sender'],
               width=3.5, height=1.0, sublabel='Senders attend to each other')
    draw_arrow(ax, (2.5, 9.3), (2.5, 8.4), COLORS['sender'], linewidth=1.2)

    # Annotation
    ax.text(5.5, 7.8, 'SenderEncoder:\nSelf-attention over\nsender tokens',
            fontsize=6, ha='left', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    # ==========================================
    # SECTION 4: Communication Encoder
    # ==========================================
    ax.text(0, 6.5, 'Communication Encoder', fontsize=10, fontweight='bold', color='gray')

    draw_block(ax, 4.5, 5.5, 'Cross-Attention (MHA)', COLORS['lr'],
               width=5.0, height=1.0, sublabel='Query: L-R tokens, Key/Value: Encoded Senders')

    draw_arrow(ax, (6.0, 9.3), (6.0, 6.1), COLORS['lr'], linewidth=1.2)
    draw_arrow(ax, (2.5, 7.2), (2.5, 6.5), COLORS['sender'])
    draw_arrow(ax, (2.5, 6.5), (3.5, 6.0), COLORS['sender'])

    # Formula annotation
    ax.text(8.5, 5.5,
            'Attn_lr←s = softmax(Q_lr K_s^T / √d) V_s\n\nL-R tokens query sender\nmemory for interaction',
            fontsize=6, ha='left', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    # ==========================================
    # SECTION 5: Relay Encoder (Response + Relay concatenated first)
    # ==========================================
    ax.text(7, 4.3, 'Relay Encoder', fontsize=9, fontweight='bold', color='gray')

    # Show concatenation of response + relay
    draw_token(ax, 8.0, 4.5, 'Resp', COLORS['relay'], width=0.6, height=0.4, fontsize=5)
    draw_token(ax, 8.7, 4.5, 'Rly', COLORS['relay'], width=0.6, height=0.4, fontsize=5)
    ax.text(9.2, 4.5, '→', fontsize=8, ha='center', va='center')
    draw_token(ax, 9.6, 4.5, 'Cat', COLORS['relay'], width=0.5, height=0.4, fontsize=5)

    draw_block(ax, 8.8, 3.5, 'Self-Attention (SAB)', COLORS['relay'],
               width=2.8, height=0.8, sublabel='Concatenated Response + Relay')
    draw_arrow(ax, (8.8, 9.3), (8.4, 4.8), COLORS['relay'], linewidth=1.2)
    draw_arrow(ax, (9.6, 4.2), (8.8, 4.0), COLORS['relay'], linewidth=0.8)

    # ==========================================
    # SECTION 6: Memory Bank
    # ==========================================
    ax.text(0, 3.8, 'Memory Bank', fontsize=10, fontweight='bold', color='gray')

    # Background
    mem_bg = FancyBboxPatch(
        (-0.3, 2.2), 6.5, 1.1,
        boxstyle="round,rounding_size=0.1",
        facecolor=COLORS['attn'], edgecolor='none', alpha=0.15
    )
    ax.add_patch(mem_bg)

    draw_token(ax, 0.5, 2.7, 'Hₛ', COLORS['sender'], width=0.6, height=0.4, fontsize=6)
    draw_token(ax, 1.4, 2.7, 'H_lr', COLORS['lr'], width=0.6, height=0.4, fontsize=6)
    draw_token(ax, 2.3, 2.7, 'H_rl', COLORS['relay'], width=0.6, height=0.4, fontsize=6)
    ax.text(3.3, 2.7, '→ Concat →', fontsize=7, ha='center', va='center')
    draw_token(ax, 4.8, 2.7, 'Memory M', COLORS['attn'], width=1.5, height=0.5, fontsize=7)

    # Arrows to memory
    draw_arrow(ax, (2.5, 7.2), (0.5, 3.0), COLORS['sender'], linewidth=0.8)
    draw_arrow(ax, (4.5, 4.9), (1.4, 3.0), COLORS['lr'], linewidth=0.8)
    draw_arrow(ax, (8.8, 3.0), (2.3, 2.8), COLORS['relay'], linewidth=0.8)

    ax.text(3.0, 2.1, 'Aggregate all encoded context', fontsize=6, color='gray', ha='center')

    # ==========================================
    # SECTION 7: Receiver Query
    # ==========================================
    ax.text(0, 1.3, 'Receiver Query', fontsize=10, fontweight='bold', color='gray')

    draw_block(ax, 3.5, 0.5, 'Cross-Attention (MHA)', COLORS['recv'],
               width=5.5, height=1.0, sublabel='Query: Receiver token, Key/Value: Memory bank')

    draw_arrow(ax, (0.2, 9.3), (0.2, 1.0), COLORS['recv'], linewidth=1.2)
    draw_arrow(ax, (0.2, 1.0), (1.2, 0.5), COLORS['recv'], linewidth=1.2)
    draw_arrow(ax, (4.8, 2.3), (4.8, 1.1), COLORS['attn'], linewidth=1.2)

    # Key equation
    ax.text(8.5, 0.5,
            'h_r = CrossAttn(Q_r, K_M, V_M)\n\nReceiver integrates all\nniche information via\nreceiver-centered attention',
            fontsize=6, ha='left', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    # ==========================================
    # SECTION 8: Output Heads
    # ==========================================
    ax.text(0, -0.8, 'Output Heads', fontsize=10, fontweight='bold', color='gray')

    # Background
    out_bg = FancyBboxPatch(
        (-0.3, -2.5), 10.5, 1.5,
        boxstyle="round,rounding_size=0.1",
        facecolor=COLORS['output'], edgecolor='none', alpha=0.1
    )
    ax.add_patch(out_bg)

    # LayerNorm
    draw_token(ax, 1.5, -1.2, 'LN', 'white', width=0.6, height=0.35, fontsize=6)
    draw_arrow(ax, (3.5, -0.1), (1.5, -1.0), COLORS['recv'])

    # Heads
    draw_block(ax, 3.5, -1.8, 'Edge₁', COLORS['output'], width=1.3, height=0.5, fontsize=7)
    draw_block(ax, 5.2, -1.8, 'Edge₂', COLORS['output'], width=1.3, height=0.5, fontsize=7)
    draw_block(ax, 6.9, -1.8, '...', COLORS['output'], width=0.8, height=0.5, fontsize=7)
    draw_block(ax, 8.5, -1.8, 'Bag Agg', COLORS['output'], width=1.4, height=0.5, fontsize=7)

    draw_arrow(ax, (1.8, -1.2), (3.0, -1.6), COLORS['output'])
    draw_arrow(ax, (1.8, -1.2), (4.7, -1.6), COLORS['output'])
    draw_arrow(ax, (1.8, -1.2), (8.0, -1.6), COLORS['output'])

    ax.text(3.5, -2.3, 'p(AAH→AIS)', fontsize=5, ha='center', color='gray')
    ax.text(5.2, -2.3, 'p(AIS→MIA)', fontsize=5, ha='center', color='gray')
    ax.text(8.5, -2.3, 'MIL pooling', fontsize=5, ha='center', color='gray')

    # ==========================================
    # Receiver path highlight
    # ==========================================
    recv_path = plt.Rectangle((-0.4, -0.3), 1.2, 12.5, fill=False,
                               edgecolor=COLORS['recv'], linestyle='--',
                               linewidth=1.5, alpha=0.7)
    ax.add_patch(recv_path)
    ax.text(-0.6, 6, 'RECEIVER\nPATH', fontsize=6, fontweight='bold',
            color=COLORS['recv'], rotation=90, ha='center', va='center')

    # ==========================================
    # Info boxes
    # ==========================================
    # Information flow
    ax.text(-1.5, 6.5,
            'Information Flow:\n'
            '1. Senders → self-attn\n'
            '2. L-R × Senders\n'
            '3. Relay self-attn\n'
            '4. Receiver ← Memory\n\n'
            'All flows terminate\n'
            'at the receiver',
            fontsize=6, va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    # Dimensions
    ax.text(-1.5, 2.5,
            'Dimensions:\n'
            'd = 128 (hidden)\n'
            'k ≤ 24 (senders)\n'
            'n ≤ 12 (L-R pairs)\n'
            'm ≤ 8 (relay)\n'
            'Heads = 4',
            fontsize=6, va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    # Novel contribution
    ax.text(8.5, -0.5,
            'Novel: Attention-Weighted\n'
            'L-R Interaction Scoring\n\n'
            'Extract Attn_r←M weights\n'
            'to quantify biological\n'
            'L-R communication strength',
            fontsize=6, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor=COLORS['recv'],
                     edgecolor=COLORS['recv'], alpha=0.15, linewidth=1.5))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['recv'], edgecolor='black', label='Receiver', alpha=0.7),
        mpatches.Patch(facecolor=COLORS['sender'], edgecolor='black', label='Senders', alpha=0.7),
        mpatches.Patch(facecolor=COLORS['lr'], edgecolor='black', label='L-R Pairs', alpha=0.7),
        mpatches.Patch(facecolor=COLORS['relay'], edgecolor='black', label='Relay', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7,
              framealpha=0.9, bbox_to_anchor=(-0.15, 1.02))

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save in multiple formats
        for fmt in ['pdf', 'png', 'svg']:
            save_path = output_path.with_suffix(f'.{fmt}')
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")

    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('docs/figures/transformer_architecture'),
        help='Output path (without extension)',
    )
    args = parser.parse_args()

    create_architecture_diagram(args.output)
    print("Done!")


if __name__ == '__main__':
    main()
