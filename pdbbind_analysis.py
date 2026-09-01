#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTR targetability study - Part 1: systematic map of nucleophilic residues
(Tyr/Cys/Ser/Thr/Lys) in ligand-binding pockets across the RCSB PDB.

用法:
    python pdbbind_analysis.py query       # 1) 采样候选 PDB ID
    python pdbbind_analysis.py process     # 2) 下载+解析+亲核残基映射 (可断点续跑)
    python pdbbind_analysis.py families    # 3) 蛋白家族分类
    python pdbbind_analysis.py summarize   # 4) 汇总统计 + 输出 CSV/JSON
    python pdbbind_analysis.py all         # 依次执行全部
"""
import os
import sys
import json
import time
import random
import re
import traceback

import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBParser, MMCIFParser

# ---------------------------------------------------------------- 路径与常量
# 注意: /mnt/agents 是 FUSE portal 文件系统 (max_read=1MB), 大文件会被驱逐,
# 因此下载缓存与中间文件放在本地磁盘 /tmp/btr_cache, 仅最终小体积输出写入输出目录。
BASE = "/mnt/agents/output/btr_study"
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
CODE_DIR = os.path.join(BASE, "code")
LOCAL_DIR = "/tmp/btr_cache"
PDB_CACHE = os.path.join(LOCAL_DIR, "pdb_files")
for d in (DATA_DIR, RESULTS_DIR, CODE_DIR, PDB_CACHE):
    os.makedirs(d, exist_ok=True)

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_PDB = "https://files.rcsb.org/download/{pid}.pdb"
DOWNLOAD_CIF = "https://files.rcsb.org/download/{pid}.cif"
ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pid}"

CANDIDATES_JSON = os.path.join(LOCAL_DIR, "candidate_ids.json")
WORKLIST_JSON = os.path.join(LOCAL_DIR, "worklist.json")
ROWS_CSV = os.path.join(LOCAL_DIR, "analysis_rows_raw.csv")   # 逐结构增量写出
FAIL_JSON = os.path.join(LOCAL_DIR, "failures.json")
FAMILIES_JSON = os.path.join(LOCAL_DIR, "families.json")
FINAL_CSV = os.path.join(DATA_DIR, "pdb_nucleophile_analysis.csv")
MINTYR_CSV = os.path.join(DATA_DIR, "min_TYR_dist_values.csv")
RESULTS_JSON = os.path.join(RESULTS_DIR, "pdb_results.json")

N_CANDIDATES = 1200       # 搜索 API 拉取的候选数
INITIAL_SAMPLE = 520      # 首轮等距/随机抽样下载尝试数
TARGET_SUCCESS = 400      # 目标成功分析数
MIN_SUCCESS = 300         # 最低成功分析数
SASA_SUBSET_N = 150       # 计算 SASA 的子集大小
SASA_MAX_ATOMS = 60000    # 超过该原子数的结构跳过 SASA(防止过慢)
DOWNLOAD_SLEEP = 0.15
FAMILY_SLEEP = 0.1

NUCLEOPHILES = ["TYR", "CYS", "SER", "THR", "LYS"]
THRESHOLDS = [5, 8, 10, 12, 15]

# 结晶添加剂/离子/缓冲剂/脂质/孤立糖等排除清单
EXCLUDE_LIGANDS = {
    "HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "FE2",
    "CU", "CU1", "CO", "NI", "CD", "HG", "SO4", "PO4", "NO3", "GOL", "EDO",
    "DMS", "PEG", "PGE", "1PE", "ACE", "ACT", "ACY", "TRS", "EPE", "MES",
    "BES", "HE3", "FMT", "CIT", "FLC", "BME", "MPD", "DIO", "BU1", "IMD",
    "NH4", "BR", "IOD", "SR", "BA", "CS", "RB", "LI", "AL", "GA", "YB", "SM",
    "EU", "TB", "PT", "AU", "AG", "MO", "WO3", "V", "CR", "OS", "IR", "PD",
    "RH", "RU", "RE", "PR", "ND", "GD", "DY", "HO", "ER", "TM", "LU", "LA",
    "CE", "TH", "U", "PU", "BEF", "ALF", "F", "SF4", "SCN", "CYN", "AZI",
    "N3", "C2O4", "MLA", "TAR", "SUC", "MAL", "GLU", "ASP", "PG4", "12P",
    "PE4", "P33", "P6G", "P4G", "P2K", "D10", "UND", "OCT", "HEX", "DM",
    "DDM", "OG", "BOG", "NG", "LMN", "C8E", "LM", "PC1", "POPC", "POPE",
    "CHOL", "CHD", "STE", "PLM", "MYR", "LRA", "SPH", "SPN", "F09", "HTG",
    "MYS", "PTY", "P2O", "PGV", "PEU", "LOP", "B7G", "GLC", "BMA", "MAN",
    "NAG", "SO2", "O", "N", "H2S", "CO2", "CO3", "UNX", "UNL", "DR6",
}

# 家族分类规则 (按优先级顺序, 对 title + pdbx_keywords 做正则匹配)
FAMILY_RULES = [
    ("GPCR", r"g[ -]?protein[- ]coupled|\bgpcr\b|adrenergic receptor|rhodopsin|"
             r"chemokine receptor|serotonin receptor|dopamine receptor|opioid receptor|"
             r"adenosine receptor|muscarinic|histamine receptor|cannabinoid receptor|"
             r"glucagon receptor|metabotropic"),
    ("Ion channel", r"ion channel|voltage[- ]gated|potassium channel|sodium channel|"
                    r"calcium channel|chloride channel|\bporin\b|trp channel|\bkcn|"
                    r"ligand[- ]gated channel|mechanosensitive channel|aquaporin"),
    ("Nuclear receptor", r"nuclear receptor|estrogen receptor|androgen receptor|"
                         r"glucocorticoid receptor|progesterone receptor|\bppar\b|"
                         r"retinoic acid receptor|retinoid x receptor|thyroid hormone receptor|"
                         r"vitamin d receptor|mineralocorticoid|liver x receptor|"
                         r"farnesoid x receptor|orphan receptor"),
    ("Antibody/Immune", r"antibod|immunoglobulin|\bfab\b|\bfv fragment|\bscfv\b|"
                        r"major histocompatibility|\bmhc\b|t[- ]cell receptor|"
                        r"interleukin|immune|complement protein|nanobod"),
    ("Kinase", r"kinase"),
    ("Phosphatase", r"phosphatase"),
    ("Protease", r"protease|peptidase|proteinase|\bcaspase\b"),
    ("Epigenetic/reader", r"bromodomain|chromodomain|histone|methyltransferase|"
                          r"demethylase|deacetylase|acetyltransferase|epigenetic|"
                          r"reader domain|\btudor domain|\bmbt domain|wd40 repeat.*histone|"
                          r"dna methyl"),
    ("Oxidoreductase", r"oxidoreductase|dehydrogenase|reductase|oxidase|peroxidase|"
                       r"cytochrome p450|\bp450\b|oxygenase|hydroxylase|"
                       r"disulfide isomerase|nitric oxide synthase"),
    ("Transporter", r"transporter|permease|symporter|antiporter|efflux|\babc\b|"
                    r"solute carrier"),
    ("Hydrolase(other)", r"hydrolase|lipase|esterase|glycosidase|amylase|nuclease|"
                         r"atpase|gtpase|helicase|phospholipase|sialidase|"
                         r"glucuronidase|lysozyme|cellulase|xylanase|amidase"),
    ("Transferase(other)", r"transferase|polymerase|synthetase|ligase"),
]

STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

# ---------------------------------------------------------------- 工具函数

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_heavy(atom):
    """判断是否为重原子(非 H/D)"""
    el = (atom.element or "").strip().upper()
    if el:
        return el not in ("H", "D")
    name = atom.get_name().strip()
    name = name.lstrip("0123456789")
    return not name[:1] in ("H", "D")


# ---------------------------------------------------------------- 1) 采样

def stage_query():
    """从 RCSB 搜索 API 获取候选 ID, 等距+种子抽样生成工作清单"""
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "exptl.method", "operator": "exact_match",
                "value": "X-RAY DIFFRACTION"}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal", "value": 3.0}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                "operator": "greater", "value": 0}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                "operator": "greater", "value": 0}},
        ]},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": N_CANDIDATES},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date",
                      "direction": "desc"}],
        },
    }
    log("Querying RCSB search API ...")
    r = requests.post(SEARCH_URL, json=query, timeout=120)
    r.raise_for_status()
    data = r.json()
    ids = [x["identifier"] for x in data["result_set"]]
    log(f"total_count={data['total_count']}, fetched {len(ids)} candidates")
    with open(CANDIDATES_JSON, "w") as f:
        json.dump(ids, f)

    random.seed(42)
    sample = random.sample(ids, min(INITIAL_SAMPLE, len(ids)))
    worklist = {"sampled": sample, "extra_round": 0}
    with open(WORKLIST_JSON, "w") as f:
        json.dump(worklist, f)
    log(f"sampled {len(sample)} ids -> worklist")


# ---------------------------------------------------------------- 2) 处理

def download_structure(pid):
    """下载 PDB, 失败则回退 mmCIF; 返回 (path, fmt) 或 (None, reason)"""
    pdb_path = os.path.join(PDB_CACHE, f"{pid}.pdb")
    cif_path = os.path.join(PDB_CACHE, f"{pid}.cif")
    if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0:
        return pdb_path, "pdb"
    if os.path.exists(cif_path) and os.path.getsize(cif_path) > 0:
        return cif_path, "cif"
    try:
        r = requests.get(DOWNLOAD_PDB.format(pid=pid), timeout=120)
        time.sleep(DOWNLOAD_SLEEP)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(pdb_path, "wb") as f:
                f.write(r.content)
            return pdb_path, "pdb"
    except Exception:
        pass
    # mmCIF 后备
    try:
        r = requests.get(DOWNLOAD_CIF.format(pid=pid), timeout=180)
        time.sleep(DOWNLOAD_SLEEP)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(cif_path, "wb") as f:
                f.write(r.content)
            return cif_path, "cif"
        return None, f"http_{r.status_code}"
    except Exception as e:
        return None, f"download_error:{type(e).__name__}"


def parse_structure(path, fmt, pid):
    if fmt == "pdb":
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)
    return parser.get_structure(pid, path)


def pick_main_ligand(model):
    """识别主配体: 排除清单外、重原子数 >=6 的 HETATM 残基中取重原子数最多者"""
    best = None
    for res in model.get_residues():
        hetflag = res.id[0]
        if hetflag == " " or hetflag == "W":
            continue
        resname = res.resname.strip()
        if resname in EXCLUDE_LIGANDS:
            continue
        heavy = [a for a in res.get_atoms() if is_heavy(a)]
        if len(heavy) < 6:
            continue
        if best is None or len(heavy) > best[2]:
            best = (res, resname, len(heavy),
                    np.array([a.get_coord() for a in heavy]))
    return best


def compute_sasa_subset(bio_structure, pid, pocket_tyr_keys):
    """用 freesasa 计算口袋 Tyr 残基侧链 SASA 之和。
    freesasa 的 PDB 读取器不能直接读 mmCIF, 故统一先用 Bio.PDB 写出临时 PDB。"""
    import freesasa
    from Bio.PDB import PDBIO
    tmp = os.path.join(LOCAL_DIR, f"_sasa_{pid}.pdb")
    try:
        io = PDBIO()
        io.set_structure(bio_structure[0])
        io.save(tmp)
        # 屏蔽 freesasa C 层 stderr 输出
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_err = os.dup(2)
        os.dup2(devnull, 2)
        try:
            structure = freesasa.Structure(tmp)
            result = freesasa.calc(structure)
        finally:
            os.dup2(old_err, 2)
            os.close(devnull)
            os.close(old_err)
        areas = result.residueAreas()
        total = 0.0
        n_found = 0
        for chain_id, resseq, icode in pocket_tyr_keys:
            chain_areas = areas.get(chain_id)
            if chain_areas is None:
                continue
            ra = chain_areas.get(str(resseq))
            if ra is None and icode.strip():
                ra = chain_areas.get(f"{resseq}{icode.strip()}")
            if ra is None:
                # 遍历匹配数字部分
                for k, v in chain_areas.items():
                    if str(resseq) == "".join(ch for ch in k if ch.isdigit() or ch == "-"):
                        ra = v
                        break
            if ra is not None:
                total += ra.sideChain
                n_found += 1
        return total, n_found
    except Exception:
        return None, 0
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def process_one(pid, sasa_budget):
    """处理单个结构, 返回 (row_dict_or_None, fail_reason_or_None, sasa_budget)"""
    path, fmt = download_structure(pid)
    if path is None:
        return None, f"download_failed:{fmt}", sasa_budget
    try:
        structure = parse_structure(path, fmt, pid)
    except Exception as e:
        return None, f"parse_failed:{type(e).__name__}", sasa_budget

    header = structure.header
    resolution = header.get("resolution")
    model = structure[0]

    lig = pick_main_ligand(model)
    if lig is None:
        return None, "no_ligand_after_filter", sasa_budget
    lig_res, lig_name, lig_nheavy, lig_coords = lig
    lig_center = lig_coords.mean(axis=0)

    # 收集蛋白链中的亲核残基
    nuc_records = {r: [] for r in NUCLEOPHILES}   # resname -> list of (min_dist, chain, resseq, icode)
    protein_chains = set()
    n_atoms_total = 0
    for chain in model:
        for res in chain:
            n_atoms_total += len(res)
            if res.id[0] != " ":
                continue
            rn = res.resname.strip()
            if rn in STD_AA:
                protein_chains.add(chain.id)
            if rn in NUCLEOPHILES:
                coords = np.array([a.get_coord() for a in res.get_atoms()
                                   if is_heavy(a)])
                if len(coords) == 0:
                    continue
                d = float(np.sqrt(((coords - lig_center) ** 2).sum(axis=1)).min())
                nuc_records[rn].append((d, chain.id, res.id[1], res.id[2]))

    row = {
        "pdb_id": pid,
        "resolution": resolution if resolution is not None else np.nan,
        "ligand_id": lig_name,
        "ligand_heavy_atoms": lig_nheavy,
        "ligand_chain": lig_res.get_parent().id,
        "n_chains": len(protein_chains),
        "structure_format": fmt,
    }
    # 阈值存在性与计数
    for resname in NUCLEOPHILES:
        dists = [x[0] for x in nuc_records[resname]]
        for t in THRESHOLDS:
            cnt = int(sum(1 for d in dists if d <= t))
            row[f"count_{resname}_{t}A"] = cnt
            row[f"has_{resname}_{t}A"] = bool(cnt > 0)
        row[f"min_{resname}_dist"] = min(dists) if dists else np.nan
    # 任意亲核残基
    all_dists = [x[0] for r in NUCLEOPHILES for x in nuc_records[r]]
    for t in THRESHOLDS:
        row[f"has_ANY_{t}A"] = bool(any(d <= t for d in all_dists))

    # SASA 子集: 口袋内 (<=10A) Tyr 残基侧链 SASA
    row["tyr_pocket_sasa"] = np.nan
    row["n_tyr_pocket_10A"] = 0
    row["sasa_computed"] = False
    pocket_tyr = [(c, s, i) for d, c, s, i in nuc_records["TYR"] if d <= 10.0]
    row["n_tyr_pocket_10A"] = len(pocket_tyr)
    if sasa_budget > 0 and n_atoms_total <= SASA_MAX_ATOMS and len(pocket_tyr) > 0:
        sasa, n_found = compute_sasa_subset(structure, pid, pocket_tyr)
        if sasa is not None:
            row["tyr_pocket_sasa"] = sasa
            row["sasa_computed"] = True
            sasa_budget -= 1
    return row, None, sasa_budget


def load_done_ids():
    if os.path.exists(ROWS_CSV):
        try:
            df = pd.read_csv(ROWS_CSV, usecols=["pdb_id"])
            return set(df["pdb_id"].tolist())
        except Exception:
            return set()
    return set()


def load_failures():
    if os.path.exists(FAIL_JSON):
        with open(FAIL_JSON) as f:
            return json.load(f)
    return {}


def count_success():
    if not os.path.exists(ROWS_CSV):
        return 0
    return sum(1 for _ in open(ROWS_CSV)) - 1


def _save_failures(failures):
    tmp = FAIL_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(failures, f, indent=1)
    os.replace(tmp, FAIL_JSON)


def stage_process():
    # 支持多进程分片并行: WORKER_INDEX / N_WORKERS 环境变量
    worker_index = int(os.environ.get("WORKER_INDEX", "0"))
    n_workers = int(os.environ.get("N_WORKERS", "1"))
    with open(CANDIDATES_JSON) as f:
        candidates = json.load(f)

    def my_shard(ids, done, failures):
        todo = [p for p in ids if p not in done and p not in failures]
        return todo[worker_index::n_workers]

    done = load_done_ids()
    failures = load_failures()
    sasa_budget = SASA_SUBSET_N
    # 已计算 SASA 的数量 (续跑时扣减预算; 每个 worker 各持一份预算)
    if os.path.exists(ROWS_CSV):
        try:
            df = pd.read_csv(ROWS_CSV, usecols=["sasa_computed"])
            sasa_budget = max(0, SASA_SUBSET_N - int(df["sasa_computed"].sum()))
        except Exception:
            pass

    with open(WORKLIST_JSON) as f:
        worklist = json.load(f)
    seen_round = worklist["extra_round"]
    todo = my_shard(worklist["sampled"], done, failures)
    log(f"worker {worker_index}/{n_workers}: {len(done)} done, {len(failures)} failed, "
        f"{len(todo)} in my shard, sasa_budget={sasa_budget}")

    processed = 0
    while True:
        for i, pid in enumerate(todo):
            try:
                row, reason, sasa_budget = process_one(pid, sasa_budget)
            except Exception as e:
                row, reason = None, f"crash:{type(e).__name__}"
                traceback.print_exc()
            if row is not None:
                rdf = pd.DataFrame([row])
                hdr = not os.path.exists(ROWS_CSV)
                rdf.to_csv(ROWS_CSV, mode="a", header=hdr, index=False)
            else:
                failures[pid] = reason
            processed += 1
            if processed % 50 == 0:
                log(f"worker{worker_index}: processed {processed}; "
                    f"success_total={count_success()}; failures_total={len(failures)}")
                _save_failures(failures)
        _save_failures(failures)

        n_succ = count_success()
        log(f"worker{worker_index}: shard finished: success={n_succ}, "
            f"failures={len(failures)}")
        if n_succ >= TARGET_SUCCESS:
            break
        if n_workers == 1 or worker_index == 0:
            # 扩展样本 (仅 worker 0 或单进程模式负责)
            with open(WORKLIST_JSON) as f:
                worklist = json.load(f)
            failures = load_failures()  # 合并其他 worker 的失败记录
            remaining = [c for c in candidates
                         if c not in worklist["sampled"] and c not in failures]
            if not remaining or worklist["extra_round"] >= 3:
                log("no more candidates or max extra rounds reached; stopping")
                break
            random.seed(42 + worklist["extra_round"] + 1)
            extra = random.sample(remaining, min(200, len(remaining)))
            worklist["sampled"].extend(extra)
            worklist["extra_round"] += 1
            tmp = WORKLIST_JSON + ".tmp"
            with open(tmp, "w") as f:
                json.dump(worklist, f)
            os.replace(tmp, WORKLIST_JSON)
            seen_round = worklist["extra_round"]
            todo = extra[worker_index::n_workers]
            log(f"extended worklist by {len(extra)} (round {worklist['extra_round']})")
        else:
            # 其他 worker 等待 worker 0 扩展工作清单
            waited = 0
            while waited < 600:
                time.sleep(10)
                waited += 10
                if count_success() >= TARGET_SUCCESS:
                    break
                with open(WORKLIST_JSON) as f:
                    wl = json.load(f)
                if wl["extra_round"] > seen_round:
                    seen_round = wl["extra_round"]
                    done = load_done_ids()
                    failures = load_failures()
                    todo = my_shard(wl["sampled"], done, failures)
                    log(f"worker{worker_index}: picked up round {seen_round}, "
                        f"{len(todo)} new in shard")
                    break
            else:
                log(f"worker{worker_index}: timeout waiting for extension; exit")
                break
            if not todo:
                if count_success() >= TARGET_SUCCESS:
                    break
                continue

    log(f"worker{worker_index} complete: success={count_success()}")


# ---------------------------------------------------------------- 3) 家族分类

def classify_family(title, keywords):
    text = f"{title or ''} {keywords or ''}".lower()
    for fam, pattern in FAMILY_RULES:
        if re.search(pattern, text):
            return fam
    return "Other"


def stage_families():
    df = pd.read_csv(ROWS_CSV, usecols=["pdb_id"])
    pdb_ids = df["pdb_id"].tolist()
    if os.path.exists(FAMILIES_JSON):
        with open(FAMILIES_JSON) as f:
            fams = json.load(f)
    else:
        fams = {}
    n_new = 0
    for i, pid in enumerate(pdb_ids):
        if pid in fams:
            continue
        try:
            r = requests.get(ENTRY_URL.format(pid=pid), timeout=60)
            time.sleep(FAMILY_SLEEP)
            if r.status_code == 200:
                d = r.json()
                title = (d.get("struct") or {}).get("title", "")
                kw = (d.get("struct_keywords") or {}).get("pdbx_keywords", "")
                fams[pid] = {"family": classify_family(title, kw),
                             "title": title, "keywords": kw}
            else:
                fams[pid] = {"family": "Other", "title": "",
                             "keywords": "", "api_status": r.status_code}
        except Exception as e:
            fams[pid] = {"family": "Other", "title": "", "keywords": "",
                         "api_status": f"error:{type(e).__name__}"}
        n_new += 1
        if n_new % 50 == 0:
            log(f"families: fetched {n_new} new (total {len(fams)}/{len(pdb_ids)})")
            with open(FAMILIES_JSON, "w") as f:
                json.dump(fams, f)
    with open(FAMILIES_JSON, "w") as f:
        json.dump(fams, f)
    log(f"families stage complete: {len(fams)} entries")


# ---------------------------------------------------------------- 4) 汇总

def stage_summarize():
    from scipy import stats

    df = pd.read_csv(ROWS_CSV)
    # 多 worker 并行时可能产生极少量重复行, 按 pdb_id 去重
    df = df.drop_duplicates(subset="pdb_id", keep="first")
    with open(FAMILIES_JSON) as f:
        fams = json.load(f)
    df["family"] = df["pdb_id"].map(lambda p: fams.get(p, {}).get("family", "Other"))

    # 列排序: 基本信息 -> family -> has_* -> count_* -> min_dist -> sasa
    base_cols = ["pdb_id", "resolution", "ligand_id", "ligand_heavy_atoms",
                 "ligand_chain", "family", "n_chains", "structure_format"]
    has_cols = [f"has_{r}_{t}A" for r in NUCLEOPHILES for t in THRESHOLDS]
    any_cols = [f"has_ANY_{t}A" for t in THRESHOLDS]
    count_cols = [f"count_{r}_{t}A" for r in NUCLEOPHILES for t in THRESHOLDS]
    dist_cols = [f"min_{r}_dist" for r in NUCLEOPHILES]
    sasa_cols = ["tyr_pocket_sasa", "n_tyr_pocket_10A", "sasa_computed"]
    df = df[base_cols + has_cols + any_cols + count_cols + dist_cols + sasa_cols]
    df.to_csv(FINAL_CSV, index=False)
    log(f"wrote {FINAL_CSV} ({len(df)} rows)")

    n = len(df)
    results = {
        "n_complexes": int(n),
        "thresholds_A": THRESHOLDS,
        "nucleophiles": NUCLEOPHILES,
        "prevalence_percent": {},
        "chi2_8A_vs_10A": {},
        "min_TYR_dist": {},
        "family_prevalence_8A": {},
        "sasa_subset": {},
    }
    # 每个阈值 x 每种亲核残基 prevalence
    for r in NUCLEOPHILES + ["ANY"]:
        results["prevalence_percent"][r] = {
            str(t): round(100.0 * df[f"has_{r}_{t}A"].mean(), 2) for t in THRESHOLDS}
    # 8A vs 10A 卡方检验 (注: 同一队列的两个阈值, 非独立样本, 仅作描述性检验)
    for r in NUCLEOPHILES:
        pos8 = int(df[f"has_{r}_8A"].sum()); neg8 = n - pos8
        pos10 = int(df[f"has_{r}_10A"].sum()); neg10 = n - pos10
        chi2, p, _, _ = stats.chi2_contingency([[pos8, neg8], [pos10, neg10]])
        results["chi2_8A_vs_10A"][r] = {
            "prev_8A": round(100.0 * pos8 / n, 2),
            "prev_10A": round(100.0 * pos10 / n, 2),
            "chi2": round(float(chi2), 4), "p_value": float(p),
            "note": "same cohort at two thresholds (non-independent); descriptive only"}
    # min Tyr 距离分布
    d = df["min_TYR_dist"].dropna().values
    bins = np.arange(0, 30.5, 0.5)
    hist, edges = np.histogram(d, bins=bins)
    qs = np.percentile(d, [0, 5, 25, 50, 75, 95, 100])
    results["min_TYR_dist"] = {
        "n_with_TYR": int(len(d)),
        "quantiles": {"q0": float(qs[0]), "q5": float(qs[1]), "q25": float(qs[2]),
                      "q50": float(qs[3]), "q75": float(qs[4]),
                      "q95": float(qs[5]), "q100": float(qs[6])},
        "histogram": {"bin_edges": [round(float(x), 2) for x in edges],
                      "counts": [int(x) for x in hist]},
    }
    # 简单双峰检测: 对直方图做平滑后找峰
    from scipy.signal import find_peaks
    smooth = np.convolve(hist, np.ones(3) / 3.0, mode="same")
    peaks, props = find_peaks(smooth, prominence=max(1.0, 0.05 * smooth.max()))
    centers = (edges[:-1] + edges[1:]) / 2
    results["min_TYR_dist"]["peak_centers_A"] = [round(float(centers[p]), 2) for p in peaks]
    results["min_TYR_dist"]["bimodal"] = bool(len(peaks) >= 2)

    # min_TYR_dist 全量值单独存 CSV (供后续绘图)
    pd.DataFrame({"pdb_id": df.loc[df["min_TYR_dist"].notna(), "pdb_id"],
                  "min_TYR_dist": d}).to_csv(MINTYR_CSV, index=False)

    # 家族 prevalence 表 (8A)
    fam_table = {}
    for fam, g in df.groupby("family"):
        entry = {"n": int(len(g))}
        for r in NUCLEOPHILES:
            entry[f"{r}_prev_8A"] = round(100.0 * g[f"has_{r}_8A"].mean(), 2)
        entry["ANY_prev_8A"] = round(100.0 * g["has_ANY_8A"].mean(), 2)
        entry["TYR_prev_10A"] = round(100.0 * g["has_TYR_10A"].mean(), 2)
        fam_table[fam] = entry
    results["family_prevalence_8A"] = dict(
        sorted(fam_table.items(), key=lambda kv: -kv[1]["TYR_prev_8A"]))

    # SASA 子集统计
    sasa = df[df["sasa_computed"]]
    results["sasa_subset"] = {
        "n_computed": int(len(sasa)),
        "tyr_pocket_sasa_mean": float(sasa["tyr_pocket_sasa"].mean()) if len(sasa) else None,
        "tyr_pocket_sasa_median": float(sasa["tyr_pocket_sasa"].median()) if len(sasa) else None,
        "tyr_pocket_sasa_quantiles": {
            "q25": float(sasa["tyr_pocket_sasa"].quantile(0.25)) if len(sasa) else None,
            "q75": float(sasa["tyr_pocket_sasa"].quantile(0.75)) if len(sasa) else None},
        "n_with_zero_pocket_tyr_sasa": int((sasa["tyr_pocket_sasa"] < 1.0).sum()) if len(sasa) else 0,
    }

    # 失败统计
    if os.path.exists(FAIL_JSON):
        with open(FAIL_JSON) as f:
            failures = json.load(f)
        from collections import Counter
        reasons = Counter(v.split(":")[0] for v in failures.values())
        results["failures"] = {"total": len(failures), "by_reason": dict(reasons)}

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {RESULTS_JSON}")

    # 中间文件(体积小)也复制到输出 data/ 目录, 便于复现与检查
    import shutil
    for src, name in [(CANDIDATES_JSON, "candidate_ids.json"),
                      (WORKLIST_JSON, "worklist.json"),
                      (FAIL_JSON, "failures.json"),
                      (FAMILIES_JSON, "families.json"),
                      (ROWS_CSV, "analysis_rows_raw.csv")]:
        if os.path.exists(src):
            dst = os.path.join(DATA_DIR, name)
            shutil.copyfile(src, dst)
            log(f"copied {name} -> {dst} ({os.path.getsize(dst)} bytes)")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("query", "all"):
        stage_query()
    if stage in ("process", "all"):
        stage_process()
    if stage in ("families", "all"):
        stage_families()
    if stage in ("summarize", "all"):
        stage_summarize()
    log("done.")
