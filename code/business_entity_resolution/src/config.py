"""
Central Configuration for Amazon ML Challenge 2026 - Business Entity Resolution Pipeline
All parameters, file paths, blocking rules, feature toggles, and model hyperparameters live here.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set

@dataclass
class PipelineConfig:
    # -------------------------------------------------------------------------
    # Directory & File Paths (strictly tab-separated TSV)
    # -------------------------------------------------------------------------
    project_root: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    data_dir: str = "dataset"
    train_dir: str = "dataset/train"
    test_dir: str = "dataset/test"
    output_dir: str = "output"
    model_dir: str = "models"
    
    # Filenames
    train_s1_file: str = "train_source1.tsv"
    train_s2_file: str = "train_source2.tsv"
    train_s3_file: str = "train_source3.tsv"
    train_gt_file: str = "train_ground_truth.tsv"
    
    test_s1_file: str = "test_source1.tsv"
    test_s2_file: str = "test_source2.tsv"
    test_s3_file: str = "test_source3.tsv"
    
    output_matching_file: str = "matching_results.tsv"
    output_candidate_file: str = "candidate_pairs.tsv"
    
    # -------------------------------------------------------------------------
    # Normalization Settings
    # -------------------------------------------------------------------------
    # Bidirectional legal suffix mapping table (English, French, Hindi)
    legal_suffix_map: Dict[str, str] = field(default_factory=lambda: {
        # English / General
        "corp": "corporation", "corporation": "corporation",
        "inc": "incorporated", "incorporated": "incorporated",
        "ltd": "limited", "limited": "limited",
        "pvt": "private", "private": "private",
        "co": "company", "company": "company",
        "llc": "llc", "l.l.c.": "llc",
        "llp": "llp", "l.l.p.": "llp",
        "pllc": "pllc", "p.l.l.c.": "pllc",
        "pc": "pc", "p.c.": "pc",
        "plc": "plc", "p.l.c.": "plc",
        # French (for France test entities)
        "sarl": "sarl", "s.a.r.l.": "sarl",
        "sas": "sas", "s.a.s.": "sas",
        "sasu": "sasu", "s.a.s.u.": "sasu",
        "sa": "sa", "s.a.": "sa",
        "sci": "sci", "s.c.i.": "sci",
        "eurl": "eurl", "e.u.r.l.": "eurl",
        "snc": "snc", "s.n.c.": "snc",
        "ste": "societe", "societe": "societe", "société": "societe",
        # Indian Languages (Hindi / Devanagari)
        "प्राइवेट लिमिटेड": "private limited",
        "प्रा. लि.": "private limited",
        "प्रा लि": "private limited",
        "लिमिटेड": "limited",
        "लि.": "limited",
        "एलएलपी": "llp",
        "कंपनी": "company",
        "कम्पनी": "company",
    })
    
    landmark_keywords: List[str] = field(default_factory=lambda: [
        "near", "opp", "opposite", "behind", "b/h", "beside",
        "adjacent", "next to", "in front of", "close to"
    ])
    
    name_stopwords: Set[str] = field(default_factory=lambda: {
        "inc", "incorporated", "corp", "corporation", "ltd", "limited",
        "pvt", "private", "co", "company", "llc", "llp", "the", "and", "of",
        "services", "solutions", "enterprises", "group", "holdings", "management"
    })
    
    addr_stopwords: Set[str] = field(default_factory=lambda: {
        "road", "rd", "street", "st", "lane", "ln", "avenue", "ave", "floor",
        "near", "opp", "opposite", "behind", "flat", "plot", "no", "co", "c", "o",
        "dr", "drive", "way", "blvd", "boulevard", "h", "house", "shop", "block",
        "bldg", "building", "apt", "apartment", "unit", "suite", "north", "south",
        "east", "west", "new", "city"
    })

    # -------------------------------------------------------------------------
    # Blocking Hyperparameters
    # -------------------------------------------------------------------------
    max_candidates_per_entity: int = 40
    max_block_token_frequency: int = 350
    min_token_len: int = 3

    # -------------------------------------------------------------------------
    # Model Hyperparameters (XGBoost - Apache 2.0 License)
    # -------------------------------------------------------------------------
    n_estimators: int = 120
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    scale_pos_weight: float = 35.0
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("/kaggle") else "cpu"
    
    # -------------------------------------------------------------------------
    # Decision Threshold & Global Consistency Post-Processing
    # -------------------------------------------------------------------------
    decision_threshold: float = 0.940
    use_global_consistency: bool = True
    
    # -------------------------------------------------------------------------
    # Hardware & Batching
    # -------------------------------------------------------------------------
    batch_size: int = 25000
    verbose: bool = True

CONFIG = PipelineConfig()
