"""
CreditLens — Dataset Loader
Loads loans.parquet into memory once at startup for fast row access.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional
import random

import pandas as pd
from loguru import logger

PARQUET_PATH = Path(__file__).parent / "loans.parquet"


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        logger.warning("loans.parquet not found — running data generation pipeline...")
        from creditlens.data.generate import run_pipeline
        run_pipeline()
    df = pd.read_parquet(PARQUET_PATH)
    logger.info(f"Dataset loaded: {len(df)} applicants")
    return df


def sample_applicants(
    n: int,
    fraud_count: int = 0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Sample n applicants from the dataset, ensuring fraud_count fraudsters are included.
    Returns a fresh shuffled DataFrame for each episode.
    """
    if seed is not None:
        random.seed(seed)

    df = load_dataset()
    fraud_df = df[df["is_fraud"] == True]
    clean_df = df[df["is_fraud"] == False]

    fraud_sample = fraud_df.sample(n=min(fraud_count, len(fraud_df)), random_state=seed)
    clean_needed = n - len(fraud_sample)
    clean_sample = clean_df.sample(n=min(clean_needed, len(clean_df)), random_state=seed)

    combined = pd.concat([fraud_sample, clean_sample]).sample(frac=1, random_state=seed).reset_index(drop=True)
    # Assign realistic applicant identifiers and banking dossier fields
    combined = combined.copy()
    
    names_pool = [
        "Priya Sharma", "Marcus Vance", "Elena Rostova", "Sophia Chen",
        "David K. Miller", "Carlos Mendoza", "Aisha Al-Mansoor", "Liam O'Connor",
        "Mei-Ling Wang", "Kwame Mensah", "Rithika Rajinikanth", "Benjamin Hayes",
        "Fatima Zahra", "Lucas Silva", "Amara Okafor", "Vikram Malhotra",
        "Rachel Green", "James Wilson", "Ananya Deshmukh", "Alexander Wright"
    ]
    employers_pool = [
        ("CloudScale Software", "Information Technology", "Permanent (Full-Time W-2)", "Elevated (Tech Layoff Index: 68)"),
        ("Metro Health System", "Healthcare & Medical", "Permanent (Full-Time W-2)", "Low (Essential Healthcare)"),
        ("Dept of Transportation", "Government & Civil Service", "Permanent (Tenured Public)", "Low (Recession-Immune)"),
        ("Apex FinTech Group", "Banking & Financial Services", "Permanent (Full-Time W-2)", "Moderate (Market Sensitive)"),
        ("Global Logistics Corp", "Logistics & Transport", "Contract / 1099", "Moderate (Fuel/Trade Cyclical)"),
        ("NextGen AI Labs", "Information Technology", "Contract (12-Mo Term)", "Elevated (Startup Runway Risk)"),
        ("BioPharma Research", "Pharmaceuticals", "Permanent (Full-Time W-2)", "Low (Non-Cyclical R&D)"),
        ("Omni Retail Brands", "Retail & Hospitality", "Probationary (6-Mo)", "Elevated (Consumer Discretionary)")
    ]

    applicant_ids = []
    applicant_names = []
    masked_ids = []
    emails = []
    employer_names = []
    work_sectors = []
    employment_types = []
    layoff_risks = []
    bank_balances = []

    for i in range(len(combined)):
        row_seed = (seed or 42) + i * 17
        app_id = f"EP_{i:03d}" if seed == 42 else f"APP-{(seed or 100) % 9000 + 1000}-{i+1:02d}"
        applicant_ids.append(app_id)
        
        name = names_pool[row_seed % len(names_pool)]
        applicant_names.append(name)
        
        masked_id = f"AADHAR-XXXX-XXXX-{(row_seed * 37) % 9000 + 1000}"
        masked_ids.append(masked_id)
        
        clean_name = name.lower().replace(" ", ".").replace("'", "").replace(".", "")
        emails.append(f"{clean_name}@verified-identity.org")
        
        emp_name, sector, emp_type, layoff = employers_pool[row_seed % len(employers_pool)]
        employer_names.append(emp_name)
        work_sectors.append(sector)
        employment_types.append(emp_type)
        layoff_risks.append(layoff)
        
        inc = float(combined.iloc[i].get("income", 60000.0))
        # Realistic liquid buffer: 1.5 to 5 months of income
        bal = round(max(3500.0, (inc / 12) * (1.5 + ((row_seed % 35) / 10))), 2)
        bank_balances.append(bal)

    combined["applicant_id"] = applicant_ids
    combined["applicant_name"] = applicant_names
    combined["masked_id"] = masked_ids
    combined["email"] = emails
    combined["employer_name"] = employer_names
    combined["work_sector"] = work_sectors
    combined["employment_type"] = employment_types
    combined["layoff_risk"] = layoff_risks
    combined["bank_balance"] = bank_balances

    return combined
