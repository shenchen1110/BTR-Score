#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regen_final_figs.py - Regenerate final figures after final-check fixes.
  D2: in-figure labels use exact CSV/JSON values rounded to 1 dp
      (fig1a, figS1a: 57.5->57.6, 62.5->62.4, 92.5->92.4)
  D3: fig1a legend moved above the axes (no overlap with bar labels)
  D4: figS2 uses constrained_layout so axis labels are not clipped
Outputs (300 dpi PNG + PDF, 7.2 in wide):
  figures/final/fig1_pdb_cohort.png/.pdf
  figures/final/figS1_pdb_extended.png/.pdf
  figures/final/figS2_components.png/.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, FIGDIR = os.path.join(BASE, 'data'), os.path.join(BASE, 'figures', 'final')
PALETTE = ['#4A6FA5', '#C08A3E', '#8B9D77', '#A65F4B', '#7A8B99', '#C4A77D', '#5C7A99']
NUCS = ['TYR', 'CYS', 'SER', 'THR', 'LYS']
THRESHOLDS = [5, 8, 10, 12, 15]

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8, 'figure.facecolor': 'white'})

pdb = pd.read_csv(f'{DATA}/pdb_nucleophile_analysis.csv')
scores = pd.read_csv(f'{DATA}/btr_protein_scores.csv')

def prev(nuc, thr):
    return pdb[f'has_{nuc}_{thr}A'].mean() * 100

def panel_label(ax, letter):
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='bottom', ha='left')

def finish(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{FIGDIR}/{stem}.{ext}', dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------- fig1
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))

ax = axes[0, 0]
x = np.arange(len(NUCS)); w = 0.38
v8 = [prev(n, 8) for n in NUCS]; v10 = [prev(n, 10) for n in NUCS]
b1 = ax.bar(x - w/2, v8, w, color=PALETTE[0], label='8 Å threshold')
b2 = ax.bar(x + w/2, v10, w, color=PALETTE[1], label='10 Å threshold')
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f'{r.get_height():.1f}', (r.get_x() + r.get_width()/2, r.get_height()),
                    ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(NUCS)
ax.set_ylabel('Prevalence (%)'); ax.set_ylim(0, 88)
ax.grid(False); sns.despine(ax=ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
          frameon=False, fontsize=8, handlelength=1.4, columnspacing=1.2)
panel_label(ax, 'A')

ax = axes[0, 1]
dist = pd.read_csv(f'{DATA}/min_TYR_dist_values.csv')['min_TYR_dist']
ax.hist(dist, bins=np.arange(0, 48.5, 0.5), color=PALETTE[0], edgecolor='white', linewidth=0.2)
ax.axvline(8, color=PALETTE[3], linestyle='--', linewidth=1)
ax.text(8.6, ax.get_ylim()[1]*0.98, '8 Å', color=PALETTE[3], fontsize=8, va='top')
ax.set_xlabel('Minimum Tyr–ligand distance (Å)'); ax.set_ylabel('Complexes')
ax.grid(False); sns.despine(ax=ax)
panel_label(ax, 'B')

ax = axes[1, 0]
for i, n in enumerate(NUCS):
    ax.plot(THRESHOLDS, [prev(n, t) for t in THRESHOLDS], marker='o', markersize=3.5,
            color=PALETTE[i], label=n, linewidth=1.4)
ax.plot(THRESHOLDS, [prev('ANY', t) for t in THRESHOLDS], marker='s', markersize=3.5,
        color=PALETTE[4], linestyle='--', label='Any', linewidth=1.4)
ax.axvline(8, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('Distance threshold (Å)'); ax.set_ylabel('Prevalence (%)')
ax.set_xticks(THRESHOLDS); ax.set_ylim(0, 105)
ax.grid(False); sns.despine(ax=ax)
ax.legend(frameon=False, fontsize=8, loc='center right')
panel_label(ax, 'C')

ax = axes[1, 1]
sasa = pdb.loc[pdb['sasa_computed'] == True, 'tyr_pocket_sasa']
ax.hist(sasa, bins=np.linspace(0, 250, 26), color=PALETTE[2], edgecolor='white', linewidth=0.3)
med = sasa.median()
ax.axvline(med, color=PALETTE[3], linestyle='--', linewidth=1)
ax.text(med + 6, ax.get_ylim()[1]*0.98, f'median {med:.1f} Å²', color=PALETTE[3], fontsize=8, va='top')
ax.set_xlabel('Pocket Tyr side-chain SASA (Å²)'); ax.set_ylabel('Complexes')
ax.grid(False); sns.despine(ax=ax)
panel_label(ax, 'D')

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.88, hspace=0.55, wspace=0.28)
finish(fig, 'fig1_pdb_cohort')

# ---------------------------------------------------------------- figS1
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))

ax = axes[0, 0]
hm = pd.DataFrame({t: [prev(n, t) for n in NUCS + ['ANY']] for t in THRESHOLDS},
                  index=NUCS + ['ANY'])
cmap = LinearSegmentedColormap.from_list('btr', ['#F5EBDD', '#C4A77D', '#A65F4B'])
annot = hm.applymap(lambda v: f'{v:.1f}')
sns.heatmap(hm, annot=annot, fmt='', cmap=cmap, vmin=0, vmax=100, ax=ax,
            cbar_kws={'label': 'Prevalence (%)'}, linewidths=0.4, linecolor='white',
            annot_kws={'fontsize': 7.5})
ax.set_xlabel('Distance threshold (Å)'); ax.set_ylabel('')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.grid(False)
panel_label(ax, 'A')

ax = axes[0, 1]
ax.hist(pdb['ligand_heavy_atoms'], bins=np.arange(5, 92, 2), color=PALETTE[0],
        edgecolor='white', linewidth=0.2)
ax.set_xlabel('Ligand heavy atoms'); ax.set_ylabel('Complexes')
ax.grid(False); sns.despine(ax=ax)
panel_label(ax, 'B')

ax = axes[1, 0]
ax.hist(pdb['resolution'], bins=35, color=PALETTE[2], edgecolor='white', linewidth=0.2)
rmed = pdb['resolution'].median()
ax.axvline(rmed, color=PALETTE[3], linestyle='--', linewidth=1)
ax.text(rmed + 0.08, ax.get_ylim()[1]*0.98, f'median {rmed:.2f} Å', color=PALETTE[3],
        fontsize=8, va='top')
ax.set_xlabel('Resolution (Å)'); ax.set_ylabel('Complexes')
ax.grid(False); sns.despine(ax=ax)
panel_label(ax, 'C')

ax = axes[1, 1]
cnt = pdb['count_TYR_8A'].clip(upper=5).value_counts().sort_index()
labels = ['0', '1', '2', '3', '4', '5+']
bars = ax.bar(labels, [cnt.get(i, 0) for i in range(6)], color=PALETTE[5])
for r in bars:
    ax.annotate(f'{int(r.get_height())}', (r.get_x() + r.get_width()/2, r.get_height()),
                ha='center', va='bottom', fontsize=8)
ax.set_xlabel('Tyr residues within 8 Å of ligand'); ax.set_ylabel('Complexes')
ax.set_ylim(0, cnt.max()*1.12)
ax.grid(False); sns.despine(ax=ax)
panel_label(ax, 'D')

fig.tight_layout(h_pad=2.2)
finish(fig, 'figS1_pdb_extended')

# ---------------------------------------------------------------- figS2
comps = [('nps', 'NPS — nucleophile proximity'), ('nas', 'NAS — nucleophile accessibility'),
         ('scs', 'SCS — structural confidence'), ('lcs', 'LCS — local chemistry')]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
for ax, (col, name), letter in zip(axes.flat, comps, 'ABCD'):
    vals = scores[col]
    ax.hist(vals, bins=np.linspace(0, 1, 21), color=PALETTE['ABCD'.index(letter)],
            edgecolor='white', linewidth=0.3)
    m = vals.mean()
    ax.axvline(m, color='#555555', linestyle='--', linewidth=1)
    ax.text(m - 0.03, ax.get_ylim()[1]*0.97, f'mean {m:.2f}', color='#555555',
            fontsize=8, va='top', ha='right')
    ax.set_xlabel(f'{name} (n={len(vals)})')
    ax.set_ylabel('Proteins')
    ax.set_xlim(-0.05, 1.08)
    ax.grid(False); sns.despine(ax=ax)
    panel_label(ax, letter)
finish(fig, 'figS2_components')

print('figures regenerated')
