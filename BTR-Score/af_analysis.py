#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
af_analysis.py — BTR 可靶向性研究 第二部分
人类膜蛋白质组 (AlphaFold 预测结构) 口袋 + 表面亲核残基图谱 + 癌症表达整合
=====================================================================================
流程:
  1) UniProt API 下载人类 reviewed 膜蛋白质组 (KW-0472 OR cc_subcellular_location:membrane)
  2) 分层随机抽样 (random_state=42), 目标 220 个蛋白
  3) AlphaFold DB API 获取 pdbUrl 并下载模型 (sleep 0.3 限速)
  4) 结构质量 (CA B-factor = pLDDT) + freesasa 表面亲核残基 (SASA > 10 A^2)
  5) LIGSITE 式网格口袋检测 (自实现, 见 detect_pockets 文档字符串)
  6) HPA 癌症表达整合 (per-gene JSON RNA cancer pTPM + v23 pathology.tsv IHC)
  7) 输出 CSV / JSON
全部输出列名与 JSON 键为英文。运行目录: 以 BASE 为根自动创建 data/ results/ code/。
"""
import os, re, io, gzip, json, time, random, datetime
import requests
import numpy as np
import pandas as pd

# freesasa 可能装在 user site-packages
import sys, site
if site.getusersitepackages() not in sys.path:
    sys.path.append(site.getusersitepackages())
import freesasa
from scipy import ndimage

# ----------------------------------------------------------------------------- 配置
BASE = os.environ.get("BTR_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA, RESULTS = os.path.join(BASE, "data"), os.path.join(BASE, "results")
MODELS = os.path.join(DATA, "af_models")
for d in (DATA, RESULTS, MODELS):
    os.makedirs(d, exist_ok=True)

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
QUERY = 'organism_id:9606 AND reviewed:true AND (keyword:KW-0472 OR cc_subcellular_location:membrane)'
FIELDS = ("accession,id,gene_names,protein_name,length,cc_subcellular_location,"
          "xref_ensembl,ft_binding,go_p,cc_disease,keyword")
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{}"
HPA_SEARCH = "https://www.proteinatlas.org/api/search_download.php?search={}&format=json&columns=g,eg,up"
HPA_GENE_JSON = "https://www.proteinatlas.org/{}.json"
HPA_PATHOLOGY_ZIP = "https://v23.proteinatlas.org/download/pathology.tsv.zip"

TARGET_SAMPLE = 220
RANDOM_SEED = 42
SLEEP = 0.3

# 口袋检测参数 (写入论文方法部分)
GRID = 1.5          # 网格步长 (A)
PROBE = 1.4         # 溶剂探针半径 (A), 加在 vdW 半径上做体素化
PAD = 4.0           # 包围盒外扩 (A)
DIR_CUTOFF = 5      # >=5/6 方向阻挡判定为口袋点
MIN_VOL, MAX_VOL = 200.0, 20000.0   # 口袋体积过滤 (A^3); >200 视为可成药候选
TOP_N = 3
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20}

NUC_RES = {"TYR", "CYS", "SER", "THR", "LYS"}
SASA_CUT = 10.0     # 表面可及阈值 (A^2)
DIST8, DIST10 = 8.0, 10.0
CANCER_PTPM_CUT = 10.0      # cancer RNA pTPM 高表达阈值
CANCER_IHC_FRAC_CUT = 0.5   # pathology High 染色患者比例阈值

CANCER_PAT = re.compile(r'cancer|tumou?r|carcinoma|onco|leukemia|lymphoma|melanoma|'
                        r'sarcoma|blastoma|glioma|myeloma|adenocarcinoma|malignan|neoplas', re.I)

session = requests.Session()


# ------------------------------------------------------------------ 1. UniProt 下载
def fetch_uniprot_all(query, fields):
    """分页下载 UniProt 查询结果 (跟随 Link: rel=next)"""
    url, params, entries, page = UNIPROT_URL, {"query": query, "size": 500,
                                               "fields": fields, "format": "json"}, [], 0
    while url:
        r = session.get(url, params=params if page == 0 else None, timeout=120)
        r.raise_for_status()
        entries.extend(r.json()["results"])
        page += 1
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        url = m.group(1) if m else None
        params = None
        if page % 5 == 0 or not url:
            print(f"  uniprot page {page}, entries {len(entries)}", flush=True)
    return entries


def parse_uniprot_entries(entries):
    rows = []
    for e in entries:
        genes = " ".join(g.get("geneName", {}).get("value", "") for g in e.get("genes", []) if "geneName" in g)
        subloc, disease, binding = "", "", []
        for c in e.get("comments", []):
            if c.get("commentType") == "SUBCELLULAR LOCATION":
                subloc += "; ".join(l.get("location", {}).get("value", "")
                                    for l in c.get("subcellularLocations", [])) + "; "
            elif c.get("commentType") == "DISEASE":
                disease += c.get("disease", {}).get("diseaseId", "") + "; "
        for f in e.get("features", []):
            if f.get("type") == "Binding site":
                pos = f.get("location", {}).get("start", {}).get("value", "")
                lig = f.get("ligand", {}).get("name", "") if f.get("ligand") else ""
                binding.append(f"{pos}:{lig}")
        rows.append(dict(
            uniprot_id=e["primaryAccession"], entry_id=e.get("uniProtkbId", ""),
            gene=genes.strip(),
            protein_name=e.get("proteinDescription", {}).get("recommendedName", {})
                           .get("fullName", {}).get("value", ""),
            length=e.get("sequence", {}).get("length"),
            subcellular_location=subloc.strip(),
            ensembl_ids=";".join(u.get("id", "") for u in e.get("uniProtKBCrossReferences", [])
                                 if u.get("database") == "Ensembl"),
            n_binding_sites=len(binding), binding_sites=";".join(binding),
            keywords=";".join(k.get("name", "") for k in e.get("keywords", [])),
            disease=disease.strip()))
    return pd.DataFrame(rows)


def fetch_transmem_counts():
    """补充抓取每个蛋白的 TRANSMEM feature 数量"""
    entries = fetch_uniprot_all(QUERY, "accession,ft_transmem")
    return {e["primaryAccession"]: sum(1 for f in e.get("features", []) if f.get("type") == "Transmembrane")
            for e in entries}


# ------------------------------------------------------------------ 2. 分层抽样
def stratified_sample(up, target=TARGET_SAMPLE, seed=RANDOM_SEED):
    txt = up["keywords"].fillna("") + ";" + up["disease"].fillna("") + ";" + up["protein_name"].fillna("")
    up = up.copy()
    up["cancer_related"] = txt.apply(lambda s: bool(CANCER_PAT.search(s)))
    elig = up[(up.length >= 100) & (up.length <= 1500)].copy()

    def strat(r):
        if r.cancer_related and r.n_transmem >= 1: return "S1_cancer_TM"
        if r.cancer_related: return "S2_cancer_nonTM"
        if r.n_transmem == 1: return "S3_singleTM"
        if r.n_transmem >= 2: return "S4_multiTM"
        return "S5_peripheral"
    elig["stratum"] = elig.apply(strat, axis=1)
    counts = elig["stratum"].value_counts()
    alloc = {"S1_cancer_TM": 40, "S2_cancer_nonTM": 26}
    rest = target - sum(alloc.values())
    others = ["S3_singleTM", "S4_multiTM", "S5_peripheral"]
    tot = sum(counts[s] for s in others)
    for s in others:
        alloc[s] = int(round(rest * counts[s] / tot))
    while sum(alloc.values()) > target: alloc[max(alloc, key=alloc.get)] -= 1
    while sum(alloc.values()) < target: alloc[min(alloc, key=alloc.get)] += 1
    sampled = [elig[elig.stratum == s].sample(n=min(n, counts[s]), random_state=seed)
               for s, n in alloc.items()]
    return pd.concat(sampled).reset_index(drop=True), elig


# ------------------------------------------------------------------ 3. AF 模型下载
def download_af_model(acc):
    """返回 (status, error)。pdbUrl 从 API 响应取 (不硬编码版本号)。"""
    pdb_path = os.path.join(MODELS, f"{acc}.pdb")
    if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 1000:
        return "ok_cached", ""
    try:
        r = session.get(AF_API.format(acc), timeout=60)
        if r.status_code == 404:
            return "no_model", "AlphaFold API 404"
        r.raise_for_status()
        preds = r.json()
        if not preds:
            return "no_model", "empty prediction list"
        pdb_url = preds[0].get("pdbUrl")
        if not pdb_url:
            return "no_model", "no pdbUrl field"
        rp = session.get(pdb_url, timeout=120)
        rp.raise_for_status()
        with open(pdb_path, "w") as f:
            f.write(rp.text)
        return "ok", ""
    except Exception as ex:
        return "api_error", str(ex)[:200]


# ------------------------------------------------------------------ 4. 结构解析 / SASA
def parse_af_pdb(pdb_path):
    """解析 AF PDB: 返回 atoms DataFrame 和 每残基 CA pLDDT (B-factor)"""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                atoms.append(dict(
                    name=line[12:16].strip(), resname=line[17:20].strip(),
                    chain=line[21].strip(), resseq=int(line[22:26]),
                    x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                    bfactor=float(line[60:66]),
                    element=line[76:78].strip() or line[12:16].strip()[0]))
    df = pd.DataFrame(atoms)
    ca = df[df.name == "CA"].set_index("resseq")["bfactor"]
    return df, ca


def residue_sasa_freesasa(pdb_path):
    """freesasa 计算每残基总 SASA -> {resseq: area}"""
    st = freesasa.Structure(pdb_path)
    res = freesasa.calc(st)
    out = {}
    for chain, d in res.residueAreas().items():
        for resseq, a in d.items():
            out[int(resseq)] = a.total
    return out


# ------------------------------------------------------------------ 5. 口袋检测
def detect_pockets(adf, grid=GRID, min_vol=MIN_VOL, max_vol=MAX_VOL, top_n=TOP_N,
                   pad=PAD, probe=PROBE, dir_cutoff=DIR_CUTOFF):
    """LIGSITE 式网格几何口袋检测 (自实现, 方法部分记录):
    1) 以 vdW+probe(1.4 A 溶剂探针) 半径将原子体素化到 grid(1.5 A) 网格;
       探针膨胀可填塞内部亚网格空隙/裂缝, 避免内部空隙网络渗滤成假口袋;
    2) 每个非占据格点沿 ±x/±y/±z 六方向扫描至包围盒边界,
       统计被蛋白占据体素阻挡的方向数 (numpy 轴向累积实现);
    3) 非占据且 >= dir_cutoff(5) 方向阻挡的格点 = 口袋点
       (开放裂口与封闭空腔均可被捕获; 溶剂可达性由 probe 膨胀保证);
    4) 26-连通域聚类 (scipy.ndimage.label);
    5) 过滤: 体积 ∈ [min_vol, max_vol] = [200, 20000] A^3,
       且聚类不触碰包围盒边界 (排除外部夹层伪影; >20000 A^3 视为未解析内部空隙网络);
    6) 按体积降序取前 top_n(3) 个。"""
    coords = adf[["x", "y", "z"]].values
    elems = adf["element"].str.upper().str[0].values
    radii = np.array([VDW.get(e, 1.70) for e in elems]) + probe
    lo = coords.min(0) - pad
    shape = tuple((np.ceil((coords.max(0) + pad - lo) / grid).astype(int) + 1).tolist())
    occ = np.zeros(shape, dtype=bool)
    idx = (coords - lo) / grid
    for i in range(len(coords)):
        c, r = idx[i], radii[i] / grid
        i0 = np.maximum(np.floor(c - r).astype(int), 0)
        i1 = np.minimum(np.ceil(c + r).astype(int) + 1, np.array(shape))
        xs = np.arange(i0[0], i1[0])[:, None, None]
        ys = np.arange(i0[1], i1[1])[None, :, None]
        zs = np.arange(i0[2], i1[2])[None, None, :]
        d2 = (xs - c[0]) ** 2 + (ys - c[1]) ** 2 + (zs - c[2]) ** 2
        occ[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] |= d2 <= r * r
    nblock = np.zeros(shape, dtype=np.int8)
    for ax in range(3):
        fwd = np.maximum.accumulate(occ, axis=ax)
        bwd = np.flip(np.maximum.accumulate(np.flip(occ, axis=ax), axis=ax), axis=ax)
        nblock += fwd.astype(np.int8) + bwd.astype(np.int8)
    pocket_pts = (~occ) & (nblock >= dir_cutoff)
    lab, nclu = ndimage.label(pocket_pts, structure=np.ones((3, 3, 3)))
    vox_vol = grid ** 3
    sx, sy, sz = shape
    out = []
    for cid in range(1, nclu + 1):
        vox = np.argwhere(lab == cid)
        vol = len(vox) * vox_vol
        if vol < min_vol or vol > max_vol:
            continue
        if (vox == 0).any() or (vox[:, 0] >= sx - 1).any() or \
           (vox[:, 1] >= sy - 1).any() or (vox[:, 2] >= sz - 1).any():
            continue  # 触碰边界 = 非封闭伪影
        out.append(dict(pocket_id=0, volume=vol, n_voxels=len(vox),
                        center=(vox.mean(0) * grid + lo).tolist()))
    out.sort(key=lambda d: -d["volume"])
    for k, d in enumerate(out[:top_n], 1):
        d["pocket_id"] = k
    return out[:top_n]


# ------------------------------------------------------------------ 6. 单蛋白分析
def analyze_protein(acc, meta):
    """返回 (summary, pocket_rows, nucleophile_rows, error)"""
    pdb_path = os.path.join(MODELS, f"{acc}.pdb")
    try:
        adf, ca = parse_af_pdb(pdb_path)
        sasa_map = residue_sasa_freesasa(pdb_path)
        res = adf.groupby("resseq").agg(resname=("resname", "first")).reset_index()
        res["sasa"] = res["resseq"].map(sasa_map).fillna(0.0)
        res["plddt"] = res["resseq"].map(ca)
        nuc = res[(res.resname.isin(NUC_RES)) & (res.sasa > SASA_CUT)]
        res_atoms = {rs: g[["x", "y", "z"]].values for rs, g in adf.groupby("resseq")}
        nuc_rows = [dict(uniprot_id=acc, resname=r.resname, resseq=int(r.resseq),
                         sasa=round(float(r.sasa), 2), plddt=round(float(r.plddt), 2))
                    for r in nuc.itertuples()]
        pocket_rows = []
        for p in detect_pockets(adf):
            ctr = np.array(p["center"])
            d8 = dict.fromkeys(NUC_RES, 0)
            d10 = dict.fromkeys(NUC_RES, 0)
            nuc_near = []
            for r in nuc.itertuples():
                dm = float(np.min(np.linalg.norm(res_atoms[r.resseq] - ctr, axis=1)))
                if dm <= DIST10:
                    nuc_near.append((dm, r.resname, int(r.resseq), float(r.sasa), float(r.plddt)))
                    if dm <= DIST8:
                        d8[r.resname] += 1
                    d10[r.resname] += 1
            tyr = [x for x in nuc_near if x[1] == "TYR"]
            pool = tyr if tyr else nuc_near          # Tyr 优先, 再按距离
            best = None
            if pool:
                b = min(pool, key=lambda x: x[0])
                best = dict(type=b[1], resseq=b[2], dist=round(b[0], 2),
                            sasa=round(b[3], 2), plddt=round(b[4], 2))
            pocket_rows.append(dict(
                uniprot_id=acc, pocket_id=p["pocket_id"], pocket_volume=round(p["volume"], 1),
                center_x=round(p["center"][0], 2), center_y=round(p["center"][1], 2),
                center_z=round(p["center"][2], 2),
                n_tyr_8A=d8["TYR"], n_cys_8A=d8["CYS"], n_ser_8A=d8["SER"],
                n_thr_8A=d8["THR"], n_lys_8A=d8["LYS"],
                n_tyr_10A=d10["TYR"], n_cys_10A=d10["CYS"], n_ser_10A=d10["SER"],
                n_thr_10A=d10["THR"], n_lys_10A=d10["LYS"],
                best_nucleophile_type=best["type"] if best else "",
                best_nuc_resseq=best["resseq"] if best else None,
                best_nuc_dist=best["dist"] if best else None,
                best_nuc_sasa=best["sasa"] if best else None,
                best_nuc_plddt=best["plddt"] if best else None,
                n_nuc_8A=sum(d8.values()), n_nuc_10A=sum(d10.values())))
        # ft_binding 对照: 注释结合位点残基 8 A 内的表面亲核残基
        bsites = [int(t.split(":")[0]) for t in str(meta.get("binding_sites") or "").split(";")
                  if t and t.split(":")[0].isdigit()]
        bcheck = []
        for bs in bsites:
            if bs not in res_atoms:
                continue
            near = []
            for r in nuc.itertuples():
                dm = float(np.min(np.linalg.norm(
                    res_atoms[r.resseq][:, None, :] - res_atoms[bs][None, :, :], axis=2)))
                if dm <= DIST8:
                    near.append((dm, r.resname))
            bcheck.append(dict(site=bs, n_nuc_8A=len(near),
                               n_tyr_8A=sum(1 for x in near if x[1] == "TYR")))
        summary = dict(uniprot_id=acc,
                       mean_plddt=round(float(ca.mean()), 2),
                       frac_plddt70=round(float((ca > 70).mean()), 4),
                       n_surface_nuc=len(nuc), n_pockets=len(pocket_rows),
                       max_pocket_volume=pocket_rows[0]["pocket_volume"] if pocket_rows else 0.0,
                       binding_sites_checked=len(bcheck),
                       binding_sites_with_tyr_8A=sum(1 for b in bcheck if b["n_tyr_8A"] > 0),
                       binding_sites_with_nuc_8A=sum(1 for b in bcheck if b["n_nuc_8A"] > 0))
        return summary, pocket_rows, nuc_rows, None
    except Exception as ex:
        return None, [], [], f"{type(ex).__name__}: {ex}"


# ------------------------------------------------------------------ 7. HPA 癌症表达
def hpa_map_ensg(accs):
    """UniProt accession -> HPA ENSG (精确匹配 Uniprot 字段); 失败兜底用基因符号"""
    ensg_map, fails = {}, {}
    for i, acc in enumerate(accs):
        try:
            r = session.get(HPA_SEARCH.format(acc), timeout=30)
            hits = json.loads(gzip.decompress(r.content).decode()) \
                if r.content[:2] == b"\x1f\x8b" else r.json()
            exact = [h for h in hits if acc in (h.get("Uniprot") or [])]
            if exact:
                ensg_map[acc] = {"hpa_gene": exact[0]["Gene"], "ensg": exact[0]["Ensembl"]}
            else:
                fails[acc] = f"no exact uniprot match ({len(hits)} hits)"
        except Exception as ex:
            fails[acc] = f"{type(ex).__name__}: {str(ex)[:100]}"
        time.sleep(SLEEP)
        if (i + 1) % 20 == 0:
            print(f"  HPA ensg map {i+1}/{len(accs)}", flush=True)
    return ensg_map, fails


def hpa_fetch_cancer_json(ensg_map):
    """per-gene JSON: RNA cancer specificity / distribution / specific pTPM / prognostics"""
    out, fails = {}, {}
    for i, (acc, m) in enumerate(ensg_map.items()):
        try:
            r = session.get(HPA_GENE_JSON.format(m["ensg"]), timeout=60)
            entry = r.json()
            if isinstance(entry, list):
                entry = entry[0]
            ptpm = entry.get("RNA cancer specific pTPM") or {}
            prog = [k for k, v in entry.items() if k.startswith("Cancer prognostics - ") and v]
            out[acc] = dict(hpa_gene=entry.get("Gene", ""), ensg=entry.get("Ensembl", ""),
                            cancer_specificity=entry.get("RNA cancer specificity", ""),
                            cancer_distribution=entry.get("RNA cancer distribution", ""),
                            cancer_specificity_score=entry.get("RNA cancer specificity score", ""),
                            max_cancer_ptpm=max([float(v) for v in ptpm.values()], default=0.0),
                            n_cancers_ptpm_listed=len(ptpm),
                            n_prognostic_significant=len(prog))
        except Exception as ex:
            fails[acc] = f"{type(ex).__name__}: {str(ex)[:100]}"
        time.sleep(SLEEP)
        if (i + 1) % 20 == 0:
            print(f"  HPA json {i+1}/{len(ensg_map)}", flush=True)
    return out, fails


def hpa_pathology_agg(zip_path):
    """下载/解析 HPA pathology.tsv -> 每 ENSG 的 IHC High/Medium 患者比例聚合"""
    if not os.path.exists(zip_path):
        r = session.get(HPA_PATHOLOGY_ZIP, timeout=600)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as f:
            patho = pd.read_csv(f, sep="\t")
    patho["total"] = patho[["High", "Medium", "Low", "Not detected"]].sum(axis=1)
    patho = patho[patho.total > 0]
    patho["frac_high"] = patho.High / patho.total
    patho["frac_highmed"] = (patho.High + patho.Medium) / patho.total
    agg = patho.groupby("Gene").agg(
        max_frac_high=("frac_high", "max"),
        max_frac_highmed=("frac_highmed", "max"),
        n_cancers_highmed_ge50=("frac_highmed", lambda s: int((s >= 0.5).sum())),
        n_cancers_tested=("Cancer", "nunique"),
        n_prognostic_fav=("prognostic - favorable", lambda s: int(s.notna().sum())),
        n_prognostic_unfav=("prognostic - unfavorable", lambda s: int(s.notna().sum())),
    ).reset_index().rename(columns={"Gene": "ensg"})
    return agg


# ------------------------------------------------------------------ main
def main():
    print("== 1. UniProt membrane proteome ==", flush=True)
    entries = fetch_uniprot_all(QUERY, FIELDS)
    up = parse_uniprot_entries(entries)
    up["n_transmem"] = up["uniprot_id"].map(fetch_transmem_counts())
    up.to_csv(os.path.join(DATA, "membrane_proteome_uniprot.csv"), index=False)
    print(f"total membrane proteins: {len(up)}")

    print("== 2. stratified sampling ==", flush=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    sample, elig = stratified_sample(up)
    sample.to_csv(os.path.join(DATA, "af_sample_list.csv"), index=False)
    print(sample["stratum"].value_counts().to_dict())

    print("== 3. AlphaFold download ==", flush=True)
    dl_log = {}
    for i, acc in enumerate(sample["uniprot_id"]):
        status, err = download_af_model(acc)
        dl_log[acc] = {"status": status, "error": err}
        if (i + 1) % 20 == 0:
            print(f"  downloaded {i+1}/{len(sample)}", flush=True)
        time.sleep(SLEEP)
    pd.DataFrame([{"uniprot_id": k, **v} for k, v in dl_log.items()]).to_csv(
        os.path.join(DATA, "af_download_log.csv"), index=False)

    print("== 4-6. structure analysis ==", flush=True)
    meta_by_acc = sample.set_index("uniprot_id").to_dict("index")
    SUMM, PR, NR, ERRS = [], [], [], []
    for i, acc in enumerate(sample["uniprot_id"]):
        s, pr, nr, err = analyze_protein(acc, meta_by_acc[acc])
        if err:
            ERRS.append({"uniprot_id": acc, "error": err})
        else:
            SUMM.append(s); PR.extend(pr); NR.extend(nr)
        if (i + 1) % 20 == 0:
            print(f"  analyzed {i+1}/{len(sample)}", flush=True)
    summ = pd.DataFrame(SUMM)
    pockets_df = pd.DataFrame(PR)
    nuc_df = pd.DataFrame(NR)

    print("== 7. HPA cancer integration ==", flush=True)
    ensg_map, ensg_fail = hpa_map_ensg(list(sample["uniprot_id"]))
    hpa_cancer, hpa_json_fail = hpa_fetch_cancer_json(ensg_map)
    patho_agg = hpa_pathology_agg(os.path.join(DATA, "pathology_v23.tsv.zip"))
    hc = pd.DataFrame([{"uniprot_id": a, **v} for a, v in hpa_cancer.items()])
    hc = hc.merge(patho_agg, on="ensg", how="left")
    hc["max_frac_high"] = hc["max_frac_high"].fillna(0.0)
    hc["cancer_high"] = (hc["max_cancer_ptpm"] > CANCER_PTPM_CUT) | \
                        (hc["max_frac_high"] >= CANCER_IHC_FRAC_CUT)
    hc.to_csv(os.path.join(DATA, "hpa_cancer_expression.csv"), index=False)

    # ---- 汇总输出 (逻辑与 notebooks 中的完全一致, 见 results/af_results.json) ----
    print("== 8. write outputs ==", flush=True)
    meta_cols = sample[["uniprot_id", "gene", "protein_name", "length", "stratum",
                        "cancer_related", "n_transmem"]].rename(
        columns={"cancer_related": "uniprot_cancer_annotation"})
    main = pockets_df.merge(summ[["uniprot_id", "mean_plddt", "frac_plddt70", "n_surface_nuc"]],
                            on="uniprot_id").merge(meta_cols, on="uniprot_id")
    main = main.merge(hc[["uniprot_id", "hpa_gene", "cancer_specificity", "cancer_distribution",
                          "max_cancer_ptpm", "max_frac_high", "n_prognostic_significant",
                          "cancer_high"]], on="uniprot_id", how="left")
    no_pocket = set(sample.uniprot_id) - set(pockets_df.uniprot_id)
    rows0 = []
    for acc in no_pocket:
        s = summ.set_index("uniprot_id").loc[acc]
        m = meta_cols.set_index("uniprot_id").loc[acc]
        h = hc.set_index("uniprot_id").to_dict("index").get(acc, {})
        rows0.append(dict(uniprot_id=acc, pocket_id=0, pocket_volume=np.nan,
                          n_tyr_8A=0, n_cys_8A=0, n_ser_8A=0, n_thr_8A=0, n_lys_8A=0,
                          n_tyr_10A=0, n_cys_10A=0, n_ser_10A=0, n_thr_10A=0, n_lys_10A=0,
                          best_nucleophile_type="", best_nuc_dist=np.nan, best_nuc_sasa=np.nan,
                          best_nuc_plddt=np.nan, mean_plddt=s.mean_plddt,
                          frac_plddt70=s.frac_plddt70, n_surface_nuc=s.n_surface_nuc,
                          gene=m.gene, protein_name=m.protein_name, length=m.length,
                          stratum=m.stratum, uniprot_cancer_annotation=m.uniprot_cancer_annotation,
                          n_transmem=m.n_transmem, hpa_gene=h.get("hpa_gene"),
                          cancer_specificity=h.get("cancer_specificity"),
                          cancer_distribution=h.get("cancer_distribution"),
                          max_cancer_ptpm=h.get("max_cancer_ptpm"),
                          max_frac_high=h.get("max_frac_high"),
                          n_prognostic_significant=h.get("n_prognostic_significant"),
                          cancer_high=h.get("cancer_high", False)))
    main = pd.concat([main, pd.DataFrame(rows0)], ignore_index=True)
    main["cancer_expression"] = main["max_cancer_ptpm"]
    main["cancer_high"] = main["cancer_high"].fillna(False).astype(bool)
    order = ["uniprot_id", "gene", "protein_name", "length", "n_transmem", "stratum",
             "mean_plddt", "frac_plddt70", "n_surface_nuc", "pocket_id", "pocket_volume",
             "n_tyr_8A", "n_cys_8A", "n_ser_8A", "n_thr_8A", "n_lys_8A",
             "n_tyr_10A", "n_cys_10A", "n_ser_10A", "n_thr_10A", "n_lys_10A",
             "best_nucleophile_type", "best_nuc_dist", "best_nuc_sasa", "best_nuc_plddt",
             "cancer_expression", "cancer_high", "cancer_specificity", "cancer_distribution",
             "max_frac_high", "n_prognostic_significant", "uniprot_cancer_annotation"]
    main = main[[c for c in order if c in main.columns]] \
             .sort_values(["uniprot_id", "pocket_id"]).reset_index(drop=True)
    main.to_csv(os.path.join(DATA, "af_nucleophile_analysis.csv"), index=False)
    nuc_out = nuc_df.merge(meta_cols[["uniprot_id", "gene"]], on="uniprot_id")[
        ["uniprot_id", "gene", "resname", "resseq", "sasa", "plddt"]] \
        .sort_values(["uniprot_id", "resseq"])
    nuc_out.to_csv(os.path.join(DATA, "af_surface_nucleophiles.csv"), index=False)

    # ---- 汇总 JSON ----
    prot = main.groupby("uniprot_id").agg(
        has_pocket=("pocket_volume", lambda s: bool(s.notna().any())),
        tyr8=("n_tyr_8A", "max"), tyr10=("n_tyr_10A", "max"),
        cancer_high=("cancer_high", "first"),
        cancer_expression=("cancer_expression", "first"),
        gene=("gene", "first"), stratum=("stratum", "first")).reset_index()
    n_tyr8 = (prot.tyr8 > 0) & prot.has_pocket
    btr = prot[n_tyr8 & prot.cancer_high]
    prevalence = {}
    for aa in ["tyr", "cys", "ser", "thr", "lys"]:
        for d in (8, 10):
            m = main.groupby("uniprot_id")[f"n_{aa}_{d}A"].max()
            prevalence[f"{aa}_{d}A"] = {"count": int((m > 0).sum()),
                                        "fraction": round(float((m > 0).mean()), 4)}
    any8 = main[["n_tyr_8A", "n_cys_8A", "n_ser_8A", "n_thr_8A", "n_lys_8A"]].sum(axis=1)
    any8 = any8.groupby(main.uniprot_id).max()
    prevalence["any_8A"] = {"count": int((any8 > 0).sum()), "fraction": round(float((any8 > 0).mean()), 4)}
    bs = summ[summ.binding_sites_checked > 0]
    results = {
        "date": datetime.date.today().isoformat(),
        "uniprot_membrane_proteome": {"total_entries": int(len(up))},
        "alphafold_download": {"attempted": len(sample),
                               "success": sum(1 for v in dl_log.values() if v["status"].startswith("ok"))},
        "structure_analysis": {"proteins_analyzed": int(len(summ)), "analysis_errors": ERRS},
        "pockets": {"proteins_with_druggable_pocket":
                    {"count": int(prot.has_pocket.sum()),
                     "fraction": round(float(prot.has_pocket.mean()), 4)}},
        "nucleophile_prevalence": prevalence,
        "cancer_integration": {"hpa_mapped": len(ensg_map), "hpa_unmapped": ensg_fail,
                               "hpa_json_failures": hpa_json_fail,
                               "cancer_high_count": int(prot.cancer_high.sum())},
        "btr_candidates": {"count": int(len(btr)),
                           "candidates": btr[["uniprot_id", "gene", "cancer_expression",
                                              "stratum"]].to_dict("records")},
        "binding_site_crosscheck": {
            "proteins_with_binding_annotation": int(len(bs)),
            "proteins_with_tyr_within_8A": int((bs.binding_sites_with_tyr_8A > 0).sum())},
    }
    with open(os.path.join(RESULTS, "af_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE. BTR candidates:", len(btr))


if __name__ == "__main__":
    main()
