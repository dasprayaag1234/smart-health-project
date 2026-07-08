"""
WHO Essential Medicines List -> PHC/CHC Primary-Care Formulary
-------------------------------------------------------------------
The WHO EML (1,738 entries) is the FULL global list - it includes cytotoxic
chemotherapy, targeted cancer therapies, hepatitis-C direct-acting antiviral
combinations, antiretroviral regimens, etc. None of that is stocked at a
rural PHC/CHC; those referrals go to district hospitals or tertiary centres.

This module filters the EML down to the subset of sections that map to what
an Indian PHC/CHC actually stocks and dispenses day-to-day (broadly aligned
with India's National List of Essential Medicines primary-care tier):
  - Access-group antibiotics (first-line, WHO AWaRe classification)
  - Antimalarials (curative)
  - Medicines for diarrhoea (ORS, zinc)
  - Antianaemia medicines (iron folic acid etc.)
  - Vitamins and minerals
  - Antihypertensives, hypoglycaemic agents & insulins (chronic disease mgmt)
  - Antiasthmatics/COPD
  - Antiseizure medicines
  - Intestinal anthelminthics
  - Antituberculosis medicines (DOTS is delivered at PHC level in India)
  - Basic dermatological anti-infectives
  - Ophthalmological anti-infectives
  - Basic antifungals
  - Vaccines
  - Diuretics (basic chronic disease support)

We keep Watch-group antibiotics OUT of the default PHC formulary (they're
typically reserve/referral-level in the Indian primary care system) but flag
them separately in case the team wants a CHC-only extended list.
"""

import pandas as pd
import os

EML_XLSX = os.path.join(os.path.dirname(__file__), "who_essential_medicines.xlsx")

PHC_RELEVANT_SECTIONS = [
    "Access group antibiotics",
    "Antimalarial medicines > Medicines for curative treatment",
    "Medicines for diarrhoea",
    "Antianaemia medicines",
    "Vitamins and minerals",
    "Antihypertensive medicines",
    "Hypoglycaemic agents",
    "Insulins",
    "Antiasthmatic and medicines for chronic obstructive pulmonary disease",
    "Antiseizure medicines",
    "Intestinal anthelminthics",
    "Antituberculosis medicines",
    "Dermatological medicines > Anti-infective medicines",
    "Diagnostic agents > Ophthalmic medicines",
    "Antifungal medicines",
    "Immunologicals > Vaccines",
    "Diuretics",
]

CHC_EXTENDED_SECTIONS = PHC_RELEVANT_SECTIONS + [
    "Watch group antibiotics",
    "Medicines affecting coagulation",
    "Antiemetic medicines",
    "General anaesthetics and oxygen > Injectable medicines",
    "Local anaesthetics",
    "Medicines used in heart failure",
]

# Rough criticality weighting: how disruptive is a stock-out of this category
# in practice (used to weight consumption rate / safety-stock priority in
# the simulator and optimizer). 3 = life-critical / time-sensitive,
# 2 = important chronic-disease management, 1 = supportive/lower urgency.
CRITICALITY_MAP = {
    "Access group antibiotics": 3,
    "Watch group antibiotics": 3,
    "Antimalarial medicines > Medicines for curative treatment": 3,
    "Medicines for diarrhoea": 3,
    "Antituberculosis medicines": 3,
    "Insulins": 3,
    "Hypoglycaemic agents": 2,
    "Antihypertensive medicines": 2,
    "Antiasthmatic and medicines for chronic obstructive pulmonary disease": 2,
    "Antiseizure medicines": 2,
    "Antianaemia medicines": 2,
    "Immunologicals > Vaccines": 3,
    "Diuretics": 2,
    "Intestinal anthelminthics": 1,
    "Vitamins and minerals": 1,
    "Dermatological medicines > Anti-infective medicines": 1,
    "Diagnostic agents > Ophthalmic medicines": 1,
    "Antifungal medicines": 1,
}


def load_formulary(path: str = EML_XLSX, tier: str = "PHC") -> pd.DataFrame:
    """tier: 'PHC' for the base primary-care list, 'CHC' for the extended list
    (adds referral-adjacent categories a CHC would stock but a PHC wouldn't)."""
    df = pd.read_excel(path)
    df = df[df["Status"] == "Added"].copy()

    sections = PHC_RELEVANT_SECTIONS if tier.upper() == "PHC" else CHC_EXTENDED_SECTIONS
    formulary = df[df["EML section"].isin(sections)].copy()

    formulary["criticality"] = formulary["EML section"].map(CRITICALITY_MAP).fillna(1).astype(int)
    formulary = formulary[["Medicine name", "EML section", "Indication", "criticality"]]
    formulary.columns = ["medicine_name", "eml_section", "indication", "criticality"]
    formulary["medicine_name"] = formulary["medicine_name"].str.strip().str.title()
    return formulary.drop_duplicates(subset="medicine_name").reset_index(drop=True)


if __name__ == "__main__":
    phc_formulary = load_formulary(tier="PHC")
    chc_formulary = load_formulary(tier="CHC")

    print(f"PHC formulary: {len(phc_formulary)} medicines")
    print(f"CHC extended formulary: {len(chc_formulary)} medicines")

    print("\nPHC formulary by criticality:")
    print(phc_formulary["criticality"].value_counts().sort_index())

    print("\nSample of PHC formulary:")
    print(phc_formulary.sample(min(15, len(phc_formulary)), random_state=1)
          [["medicine_name", "eml_section", "criticality"]].to_string(index=False))

    out_dir = os.path.dirname(__file__)
    phc_formulary.to_csv(os.path.join(out_dir, "phc_formulary.csv"), index=False)
    chc_formulary.to_csv(os.path.join(out_dir, "chc_formulary.csv"), index=False)
    print("\nSaved -> phc_formulary.csv, chc_formulary.csv")
