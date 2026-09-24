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
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")
    
    true_matches = {}
    all_s1_ids = []
    
    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        all_s1_ids.append(s1)
        raw_mids = str(row["matched_entity_ids"]).strip()
        if raw_mids:
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
