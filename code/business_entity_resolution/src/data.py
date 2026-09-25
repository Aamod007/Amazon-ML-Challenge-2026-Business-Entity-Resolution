"""
Data Handling Module
Handles strict tab-separated TSV ingestion, ground truth parsing, singleton preservation,
and entity-grouped validation splitting without information leakage.
All files are strictly read and written with explicit tab separation (sep="\\t").
"""

import os
from typing import Dict, List, Set, Tuple, Iterator, Any
import pandas as pd
try:
    from .config import CONFIG
    from .normalize import normalize_record
except ImportError:
    from config import CONFIG
    from normalize import normalize_record

def load_ground_truth(gt_path: str) -> Tuple[Dict[str, Set[str]], List[str]]:
    """
    Parse ground truth TSV into:
      - true_matches: Dict[s1_entity_id -> set of matched_entity_ids]
      - all_s1_ids: complete list of S1 entity IDs in ground truth
    Treats empty matched_entity_ids as an explicit singleton label, not missing data.
    """
    print(f"Loading ground truth from: {gt_path}")
    true_matches = {}
    all_s1_ids = []
    
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("source1_entity_id")
        match_idx = header.index("matched_entity_ids")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1 = parts[id_idx].strip()
            all_s1_ids.append(s1)
            if len(parts) > match_idx and parts[match_idx].strip():
                raw_mids = parts[match_idx].strip()
                mids = set(m.strip() for m in raw_mids.split(",") if m.strip())
            else:
                mids = set()
            true_matches[s1] = mids
            
    singletons = sum(1 for mids in true_matches.values() if len(mids) == 0)
    print(f"Parsed {len(all_s1_ids):,} S1 entities: {singletons:,} singletons ({singletons/len(all_s1_ids)*100:.2f}%)")
    return true_matches, all_s1_ids

def stream_tsv_records(tsv_path: str) -> Iterator[Tuple[str, str, str, str]]:
    """
    Stream records from a TSV file yielding (entity_id, business_name, business_address, country).
    Uses Polars for multi-threaded C++/Rust accelerated tab-separated reading.
    """
    try:
        import polars as pl
        df = pl.read_csv(
            tsv_path,
            separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"],
            null_values=[""],
            schema_overrides={"entity_id": pl.String, "business_name": pl.String, "business_address": pl.String, "country": pl.String}
        )
        for row in df.iter_rows():
            yield (
                str(row[0] or "").strip(),
                str(row[1] or "").strip(),
                str(row[2] or "").strip(),
                str(row[3] or "").strip()
            )
    except ImportError:
        with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().split("\t")
            id_col = header.index("entity_id")
            name_col = header.index("business_name")
            addr_col = header.index("business_address")
            c_col = header.index("country")
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    yield (
                        parts[id_col].strip(),
                        parts[name_col].strip(),
                        parts[addr_col].strip(),
                        parts[c_col].strip()
                    )

def load_records_by_country(tsv_path: str, target_country: str = None) -> Dict[str, Dict[str, Any]]:
    """
    Load and normalize records from TSV, optionally filtered by country string.
    Country is treated as an open string label.
    """
    records = {}
    try:
        import polars as pl
        df = pl.read_csv(
            tsv_path,
            separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"],
            schema_overrides={"entity_id": pl.String, "business_name": pl.String, "business_address": pl.String, "country": pl.String}
        )
        if target_country is not None:
            df = df.filter(pl.col("country") == target_country)
        for row in df.iter_rows():
            raw_tuple = (
                str(row[0] or "").strip(),
                str(row[1] or "").strip(),
                str(row[2] or "").strip(),
                str(row[3] or "").strip()
            )
            rec = normalize_record(raw_tuple)
            records[rec["entity_id"]] = rec
    except ImportError:
        for raw_tuple in stream_tsv_records(tsv_path):
            c = raw_tuple[3]
            if target_country is None or c == target_country:
                rec = normalize_record(raw_tuple)
                records[rec["entity_id"]] = rec
    return records

def create_official_split(
    train_dir: str,
    n_train_s1: int = 15000,
    n_val_s1: int = 4000,
    max_bg_records: int = 40000,
    random_state: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split training dataset into Train and Validation sets that strictly respect official test structure:
      1. Stratified by singleton vs non-singleton status (~5.58% singletons).
      2. Stratified by country (US ~60%, India ~40%).
      3. Strict entity-level separation: zero S1 entities in common.
      4. Independent validation candidate pool with realistic background distractors,
         enabling full end-to-end evaluation (blocking recall -> feature scoring -> global consistency).
    """
    import numpy as np
    import polars as pl
    
    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    s2_file = os.path.join(train_dir, "train_source2.tsv")
    s3_file = os.path.join(train_dir, "train_source3.tsv")

    print("\n" + "="*70)
    print("STAGE 1: CREATING LEAKAGE-FREE OFFICIAL VALIDATION SPLIT")
    print("="*70)
    
    # 1. Load ground truth
    true_matches_all, all_s1_ids = load_ground_truth(gt_file)
    
    # 2. Load S1 metadata (country)
    df_s1_meta = pl.read_csv(
        s1_file,
        separator="\t",
        columns=["entity_id", "country"],
        schema_overrides={"entity_id": pl.String, "country": pl.String}
    )
    s1_country_map = dict(zip(df_s1_meta["entity_id"].to_list(), df_s1_meta["country"].to_list()))
    
    # 3. Stratified partition: group by (is_singleton, country)
    strata = {}
    for s1 in all_s1_ids:
        is_sing = (len(true_matches_all[s1]) == 0)
        c = s1_country_map.get(s1, "Unknown")
        key = (is_sing, c)
        if key not in strata:
            strata[key] = []
        strata[key].append(s1)
        
    rng = np.random.RandomState(random_state)
    total_requested = n_train_s1 + n_val_s1
    sample_ratio = min(1.0, total_requested / max(len(all_s1_ids), 1))
    
    train_s1_ids = []
    val_s1_ids = []
    
    for (is_sing, c), ids in strata.items():
        rng.shuffle(ids)
        n_stratum_total = int(len(ids) * sample_ratio)
        n_stratum_val = int(n_stratum_total * (n_val_s1 / max(total_requested, 1)))
        n_stratum_train = n_stratum_total - n_stratum_val
        
        train_s1_ids.extend(ids[:n_stratum_train])
        val_s1_ids.extend(ids[n_stratum_train:n_stratum_train + n_stratum_val])
        
    print(f"Sampled {len(train_s1_ids):,} Train S1 entities and {len(val_s1_ids):,} Validation S1 entities.")
    train_sing_ratio = sum(1 for s in train_s1_ids if len(true_matches_all[s]) == 0) / max(len(train_s1_ids), 1)
    val_sing_ratio = sum(1 for s in val_s1_ids if len(true_matches_all[s]) == 0) / max(len(val_s1_ids), 1)
    print(f"Singleton preservation: Train={train_sing_ratio*100:.2f}%, Val={val_sing_ratio*100:.2f}% (Ground Truth ~5.58%)")
    
    # 4. Extract true targets for Train and Val
    train_targets = set()
    for s1 in train_s1_ids:
        train_targets.update(true_matches_all[s1])
        
    val_targets = set()
    for s1 in val_s1_ids:
        val_targets.update(true_matches_all[s1])
        
    print(f"Train target S2/S3 IDs: {len(train_targets):,} | Validation target S2/S3 IDs: {len(val_targets):,}")
    
    # 5. Load and normalize S1 records
    print("Normalizing Source 1 records...")
    df_s1 = pl.read_csv(s1_file, separator="\t")
    s1_train_records = {}
    s1_val_records = {}
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)
    
    for row in df_s1.filter(pl.col("entity_id").is_in(list(train_s1_set.union(val_s1_set)))).iter_rows():
        rec = normalize_record(row)
        eid = rec["entity_id"]
        if eid in train_s1_set:
            s1_train_records[eid] = rec
        elif eid in val_s1_set:
            s1_val_records[eid] = rec
            
    # 6. Load candidate pool from S2 and S3 with background distractors
    print("Loading candidate pools from Source 2 and Source 3...")
    pool_train_records = {}
    pool_val_records = {}
    
    for src_file in [s2_file, s3_file]:
        df_src = pl.read_csv(src_file, separator="\t")
        
        # Train pool records
        df_train_targets = df_src.filter(pl.col("entity_id").is_in(list(train_targets)))
        df_train_bg = df_src.head(max_bg_records // 2)
        df_train_comb = pl.concat([df_train_targets, df_train_bg]).unique(subset=["entity_id"])
        for row in df_train_comb.iter_rows():
            pool_train_records[row[0]] = normalize_record(row)
            
        # Validation pool records (take background from bottom of files to ensure distractor independence)
        df_val_targets = df_src.filter(pl.col("entity_id").is_in(list(val_targets)))
        df_val_bg = df_src.tail(max_bg_records // 2)
        df_val_comb = pl.concat([df_val_targets, df_val_bg]).unique(subset=["entity_id"])
        for row in df_val_comb.iter_rows():
            pool_val_records[row[0]] = normalize_record(row)
            
    print(f"Candidate pools constructed: Train Pool={len(pool_train_records):,} | Val Pool={len(pool_val_records):,}")
    
    train_data = {
        "s1_records": s1_train_records,
        "pool_records": pool_train_records,
        "true_matches": {s: true_matches_all[s] for s in train_s1_ids},
        "s1_ids": train_s1_ids
    }
    
    val_data = {
        "s1_records": s1_val_records,
        "pool_records": pool_val_records,
        "true_matches": {s: true_matches_all[s] for s in val_s1_ids},
        "s1_ids": val_s1_ids
    }
    
    return train_data, val_data
