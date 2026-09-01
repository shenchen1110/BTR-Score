# BTR-Score

Code and derived data for the manuscript:

**"Structural bioinformatics survey of binding-triggered release targetability prioritizes extracellular cancer targets"**
Chen Shen, Deze Li, Hao Chen, Xin Ni — Beijing Children's Hospital, Capital Medical University

Repository: https://github.com/shenchen1110/BTR-Score

## Contents
| File | Description |
|---|---|
| `pdbbind_analysis.py` | PDB ligand–complex nucleophile mapping (490 complexes, RCSB PDB) |
| `af_analysis.py` | AlphaFold human membrane proteome pocket & nucleophile analysis (220 models) |
| `btr_score_analysis.py` | BTR-Score computation, validation, prioritization, weight-sensitivity analysis |
| `regen_final_figs.py` | Figure regeneration scripts (Figs. 1–5, S1–S4) |
| `data/` | All derived result tables (CSV) |

## Requirements
Python 3.10+; packages: `biopython`, `freesasa`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `requests`.

## Data sources (all public)
RCSB PDB (https://www.rcsb.org), AlphaFold Protein Structure Database (https://alphafold.ebi.ac.uk),
UniProt Knowledgebase (https://www.uniprot.org), Human Protein Atlas v23 (https://www.proteinatlas.org).

## License
MIT (see LICENSE).
