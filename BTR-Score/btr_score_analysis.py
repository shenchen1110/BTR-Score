#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
btr_score_analysis.py — BTR 可靶向性研究 第三部分
BTR-Score 计算 + 阳性/阴性对照验证 + 靶点优先级排序 + 动态可及性分析 +
统计检验 + 出版级图表 + final_results.json
=====================================================================================
输入 (上游第一/二部分产出):
  data/af_nucleophile_analysis.csv  (561 行 蛋白-口袋 记录, 220 蛋白)
  data/pdb_nucleophile_analysis.csv (490 个 PDB 配体复合物)
  data/min_TYR_dist_values.csv
  results/pdb_results.json, results/af_results.json
  data/af_models/*.pdb (220 个 AF 模型)
输出:
  data/btr_protein_scores.csv, data/btr_target_prioritization.csv,
  data/dynamic_accessibility_records.csv
  results/validation_results.json, results/dynamic_accessibility.json,
  results/statistical_tests.json, results/figure_legends.json, results/final_results.json
  figures/fig1..fig5 (300 dpi PNG + PDF)
注: 本脚本为分析流水线的记录版; 中间文件写 /tmp/btr_cache_c/。
"""
import os, sys, json, math, time, importlib.util, warnings
import numpy as np
import pandas as pd
import requests
import freesasa
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
from Bio.PDB.vectors import calc_dihedral
from scipy import stats
import scikit_posthocs as sp
warnings.filterwarnings('ignore')

BASE = os.environ.get('BTR_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, RESULTS, FIGDIR = (os.path.join(BASE, d) for d in ('data', 'results', 'figures'))
CACHE = '/tmp/btr_cache_c'
for d in (FIGDIR, CACHE, f'{CACHE}/validation_models', f'{CACHE}/dyn_structures'):
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------ BTR-Score 定义
REACT = {'TYR': 0.9, 'CYS': 1.0, 'SER': 0.7, 'THR': 0.6, 'LYS': 0.5}
W_NPS, W_NAS, W_SCS, W_LCS = 0.35, 0.30, 0.20, 0.15

def nps_score(d):    # 距离 <=4A=1.0, >=15A=0, 线性衰减
    if d is None or pd.isna(d): return 0.0
    return float(np.clip((15.0 - d) / 11.0, 0.0, 1.0))

def nas_score(s):    # SASA >=30=1.0, <=10=0, 线性
    if s is None or pd.isna(s): return 0.0
    return float(np.clip((s - 10.0) / 20.0, 0.0, 1.0))

def scs_score_plddt(p):  # pLDDT>90=1.0, 70-90=0.5, <70=0
    if p is None or pd.isna(p): return 0.0
    return 1.0 if p > 90 else (0.5 if p >= 70 else 0.0)

def scs_score_bfactor(b_res, b_all):  # 实验结构: 链内 z 归一化 B-factor
    z = (b_res - np.mean(b_all)) / np.std(b_all)
    return (1.0 if z <= -0.5 else (0.5 if z <= 0.5 else 0.0)), float(z)

def lcs_score(ntype, d):  # 反应性 x 距离因子 (<6=1.0, <10=0.7, >=10=0.3)
    if not ntype or pd.isna(ntype): return 0.0
    rw = REACT.get(str(ntype).upper(), 0.0)
    return rw * (1.0 if d < 6 else (0.7 if d < 10 else 0.3))

def btr_components(ntype, dist, sasa, plddt):
    nps, nas = nps_score(dist), nas_score(sasa)
    scs, lcs = scs_score_plddt(plddt), lcs_score(ntype, dist)
    return nps, nas, scs, lcs, W_NPS*nps + W_NAS*nas + W_SCS*scs + W_LCS*lcs

def category(s):
    return 'high' if s >= 0.7 else ('medium' if s >= 0.4 else 'low')

# ------------------------------------------------------------------ 1. AF 队列打分
def score_af_cohort():
    af = pd.read_csv(f'{DATA}/af_nucleophile_analysis.csv')
    rows = []
    for _, r in af.iterrows():
        nt = r['best_nucleophile_type'] if isinstance(r['best_nucleophile_type'], str) else None
        nps, nas, scs, lcs, sc = btr_components(nt, r['best_nuc_dist'],
                                                r['best_nuc_sasa'], r['best_nuc_plddt'])
        rows.append((r['uniprot_id'], r['pocket_id'], nps, nas, scs, lcs, sc))
    scr = pd.DataFrame(rows, columns=['uniprot_id','pocket_id','nps','nas','scs','lcs','btr_score'])
    af_s = af.merge(scr, on=['uniprot_id','pocket_id'], how='left')
    prot = af_s.loc[af_s.groupby('uniprot_id')['btr_score'].idxmax()].copy()  # 每蛋白最优口袋
    prot['category'] = prot['btr_score'].apply(category)
    prot['expression_norm'] = prot['cancer_expression'] / prot['cancer_expression'].max()
    prot['priority_score'] = prot['btr_score'] * (0.7 + 0.3 * prot['expression_norm'])
    return prot.sort_values('priority_score', ascending=False).reset_index(drop=True)

# ------------------------------------------------------------------ 2. 对照验证
def import_afmod():
    spec = importlib.util.spec_from_file_location('afmod', f'{BASE}/code/af_analysis.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def download_af(acc, outdir, sess):
    r = sess.get(f'https://alphafold.ebi.ac.uk/api/prediction/{acc}', timeout=60)
    r.raise_for_status()
    pdb_url = r.json()[0]['pdbUrl']
    rp = sess.get(pdb_url, timeout=120); rp.raise_for_status()
    path = os.path.join(outdir, f'{acc}.pdb')
    with open(path, 'w') as f: f.write(rp.text)
    return path

def analyze_af_pdb(afmod, pdb_path):
    """与上游完全一致的口袋检测 + 最优亲核残基 (含 resseq)"""
    adf, ca = afmod.parse_af_pdb(pdb_path)
    sasa_map = afmod.residue_sasa_freesasa(pdb_path)
    res = adf.groupby('resseq').agg(resname=('resname','first')).reset_index()
    res['sasa'] = res['resseq'].map(sasa_map).fillna(0.0)
    res['plddt'] = res['resseq'].map(ca)
    nuc = res[(res.resname.isin(afmod.NUC_RES)) & (res.sasa > afmod.SASA_CUT)]
    res_atoms = {rs: g[['x','y','z']].values for rs, g in adf.groupby('resseq')}
    pockets = []
    for p in afmod.detect_pockets(adf):
        ctr = np.array(p['center']); near = []
        for r in nuc.itertuples():
            dm = float(np.min(np.linalg.norm(res_atoms[r.resseq]-ctr, axis=1)))
            if dm <= 10.0:
                near.append((dm, r.resname, int(r.resseq), float(r.sasa), float(r.plddt)))
        tyr = [x for x in near if x[1] == 'TYR']
        pool = tyr if tyr else near
        best = min(pool, key=lambda x: x[0]) if pool else None
        pockets.append(dict(pocket_id=p['pocket_id'], volume=round(p['volume'],1),
                            center=[round(c,2) for c in p['center']],
                            best=(dict(type=best[1], resseq=best[2], dist=round(best[0],2),
                                       sasa=round(best[3],2), plddt=round(best[4],2)) if best else None)))
    summary = dict(mean_plddt=round(float(ca.mean()),2),
                   frac_plddt70=round(float((ca>70).mean()),4),
                   n_residues=int(len(ca)), n_surface_nuc=int(len(nuc)))
    return summary, pockets

# ------------------------------------------------------------------ 3. 动态可及性
def rcsb_search_uniprot(sess, acc, rows=60):
    q = {"query":{"type":"group","logical_operator":"and","nodes":[
        {"type":"terminal","service":"text","parameters":{
          "attribute":"rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
          "operator":"exact_match","value":acc}},
        {"type":"terminal","service":"text","parameters":{
          "attribute":"rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
          "operator":"exact_match","value":"UniProt"}}]},
        "request_options":{"paginate":{"start":0,"rows":rows}},
        "return_type":"entry"}
    r = sess.post('https://search.rcsb.org/rcsbsearch/v2/query', json=q, timeout=60)
    r.raise_for_status()
    return [x['identifier'] for x in r.json().get('result_set', [])]

def sifts_segments(sess, pid, acc):
    r = sess.get(f'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pid.lower()}', timeout=60)
    r.raise_for_status()
    return r.json()[pid.lower()]['UniProt'][acc]['mappings']

def cif_label_auth_map(path):
    """mmCIF _atom_site: (auth_chain, label_seq_id) -> (auth_seq_id, comp)"""
    rows = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == 'loop_':
            j = i+1; cols = []
            while j < len(lines) and lines[j].startswith('_'):
                cols.append(lines[j].strip()); j += 1
            if cols and cols[0].startswith('_atom_site.'):
                names = [c.split('.')[1] for c in cols]
                need = ['label_seq_id','auth_seq_id','label_comp_id','auth_asym_id']
                idx = {n: names.index(n) for n in need}
                k = j
                while k < len(lines) and lines[k].startswith(('ATOM','HETATM')):
                    p = lines[k].split()
                    if len(p) >= len(names):
                        rows.append({n: p[idx[n]] for n in idx})
                    k += 1
                break
            i = j
        else:
            i += 1
    mp = {}
    for r in rows:
        if r['label_seq_id'] in ('.','?'): continue
        mp[(r['auth_asym_id'], int(r['label_seq_id']))] = (int(r['auth_seq_id']), r['label_comp_id'])
    return mp

class ChainOnly(Select):
    """仅保留指定链的标准残基 (apo 表示, 供跨结构可比 SASA)"""
    def __init__(self, cid): self.cid = cid
    def accept_chain(self, chain): return chain.id == self.cid
    def accept_residue(self, residue): return residue.id[0] == ' '

def chain_apo_sasa(structure, pid, chid):
    io = PDBIO(); io.set_structure(structure)
    out = f'{CACHE}/dyn_structures/{pid}_{chid}_apo.pdb'
    io.save(out, ChainOnly(chid))
    res = freesasa.calc(freesasa.Structure(out))
    return {int(rs): a.total for _, d in res.residueAreas().items() for rs, a in d.items()}

# ------------------------------------------------------------------ main 流程说明
# 实际运行按 notebook 顺序执行:
#   1) prot = score_af_cohort()                     -> btr_protein_scores.csv / btr_target_prioritization.csv
#   2) FAP Q12884 / PD-L1 Q9NZQ7 AF 模型下载 + analyze_af_pdb 打分;
#      PD-L1 网格法无口袋 -> 回退实验复合物 5J89 (BMS-202), apo 链 SASA + z-B-factor SCS;
#      阴性对照 = 队列最低分蛋白 (TRBC2, IGKV2D-30, SEC24B; score 0)
#   3) 动态可及性: EGFR Tyr585 / PDGFRA Tyr273 / EPCAM Tyr251 / FAP Tyr467 / PD-L1 Tyr56
#      RCSB UniProt 检索 -> SIFTS 映射 -> cif 下载 -> 每链 apo SASA + chi1;
#      HAVCR2 Tyr281 无实验结构覆盖, 记录为局限
#   4) 统计: Kruskal-Wallis (家族 min_TYR_dist) + Dunn-Bonferroni;
#      Fisher 精确 + 卡方 (高/低 BTR 组 cancer_high 比例)
#   5) 图 fig1-fig5 (暖色低饱和调色板, 白底, 300 dpi PNG+PDF) + figure_legends.json
#   6) final_results.json 汇总全部数字
if __name__ == '__main__':
    prot = score_af_cohort()
    print(prot['category'].value_counts())
    print('Run full pipeline interactively per section docstring; see results/final_results.json')
