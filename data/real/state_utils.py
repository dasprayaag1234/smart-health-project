"""
State/UT name harmonization across datasets.

The NFHS-5 district file, the PHC/CHC beds file, and (eventually) other
government datasets don't use consistent State/UT naming or boundaries
(e.g. Daman & Diu merged with Dadra & Nagar Haveli in 2020; spelling
variants; "NCT of Delhi" vs "Delhi"). This module provides one canonical
mapping so every loader agrees on state names before merging.
"""

STATE_NAME_MAP = {
    # NFHS name -> canonical name used across this project
    "Andaman & Nicobar Islands": "Andaman & Nicobar Islands",
    "A & N Island": "Andaman & Nicobar Islands",
    "Maharastra": "Maharashtra",
    "Maharashtra": "Maharashtra",
    "NCT of Delhi": "Delhi",
    "Delhi": "Delhi",
    "Dadra and Nagar Haveli & Daman and Diu": "Dadra and Nagar Haveli & Daman and Diu",
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli & Daman and Diu",
    "Daman & Diu": "Dadra and Nagar Haveli & Daman and Diu",
}

# States/UTs with no direct match in the beds dataset (post-2019 boundary
# changes, e.g. Ladakh split from J&K in 2019 after the beds source's
# reference year). We fall back to the parent/neighboring UT's facility
# norms rather than dropping these districts from the simulation.
FALLBACK_STATE_FOR_INFRA = {
    "Ladakh": "Jammu & Kashmir",
}


def canonical_state(name: str) -> str:
    name = name.strip()
    return STATE_NAME_MAP.get(name, name)


def infra_lookup_state(name: str) -> str:
    """Canonical name to use specifically when looking up facility
    infrastructure norms (handles the Ladakh fallback)."""
    canon = canonical_state(name)
    return FALLBACK_STATE_FOR_INFRA.get(canon, canon)
