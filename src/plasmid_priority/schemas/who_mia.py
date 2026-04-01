"""Curated WHO Medically Important Antimicrobials (MIA) class mapping.

Extracted to a standalone schema module so that both feature engineering
(``features.core``) and reporting (``reporting.external_support``) can
reference the same authoritative mapping without circular imports.
"""

from __future__ import annotations


WHO_MIA_CLASS_MAP: dict[str, dict[str, object]] = {
    "AMINOGLYCOSIDE": {
        "who_mia_category": "CIA",
        "who_mia_class": "Aminoglycosides",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "CARBAPENEM": {
        "who_mia_category": "HPCIA",
        "who_mia_class": "Carbapenems with or without inhibitors",
        "who_mapping_scope": "humans_only",
    },
    "COLISTIN": {
        "who_mia_category": "HPCIA",
        "who_mia_class": "Polymyxins",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "DIAMINOPYRIMIDINE": {
        "who_mia_category": "HIA",
        "who_mia_class": "Sulfonamides, dihydrofolate reductase inhibitors and combinations",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "FLUOROQUINOLONE": {
        "who_mia_category": "HPCIA",
        "who_mia_class": "Quinolones",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "FOSFOMYCIN": {
        "who_mia_category": "HPCIA",
        "who_mia_class": "Phosphonic acid derivatives",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "FUSIDANE": {
        "who_mia_category": "HIA",
        "who_mia_class": "Fusidanes",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "FUSIDIC ACID": {
        "who_mia_category": "HIA",
        "who_mia_class": "Fusidanes",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "GLYCOPEPTIDE": {
        "who_mia_category": "HPCIA",
        "who_mia_class": "Glycopeptides and lipoglycopeptides",
        "who_mapping_scope": "humans_only",
    },
    "GLYCYLCYCLINE": {
        "who_mia_category": "CIA",
        "who_mia_class": "Glycylcyclines",
        "who_mapping_scope": "humans_only",
    },
    "LINCOSAMIDE": {
        "who_mia_category": "HIA",
        "who_mia_class": "Lincosamides",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "MACROLIDE": {
        "who_mia_category": "CIA",
        "who_mia_class": "Macrolides",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "MONOBACTAM": {
        "who_mia_category": "CIA",
        "who_mia_class": "Monobactams",
        "who_mapping_scope": "humans_only",
    },
    "NITROFURAN": {
        "who_mia_category": "IA",
        "who_mia_class": "Nitrofuran derivatives",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "OXAZOLIDINONE": {
        "who_mia_category": "CIA",
        "who_mia_class": "Oxazolidinones",
        "who_mapping_scope": "humans_only",
    },
    "PHENICOL": {
        "who_mia_category": "HIA",
        "who_mia_class": "Amphenicols",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "PHOSPHONIC ACID": {
        "who_mia_category": "HPCIA",
        "who_mia_class": "Phosphonic acid derivatives",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "PLEUROMUTILIN": {
        "who_mia_category": "IA",
        "who_mia_class": "Pleuromutilins",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "POLYMYXIN": {
        "who_mia_category": "HPCIA",
        "who_mia_class": "Polymyxins",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "QUINOLONE": {
        "who_mia_category": "HPCIA",
        "who_mia_class": "Quinolones",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "RIFAMYCIN": {
        "who_mia_category": "CIA",
        "who_mia_class": "Ansamycins",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "SULFONAMIDE": {
        "who_mia_category": "HIA",
        "who_mia_class": "Sulfonamides, dihydrofolate reductase inhibitors and combinations",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "TETRACYCLINE": {
        "who_mia_category": "HIA",
        "who_mia_class": "Tetracyclines",
        "who_mapping_scope": "both_humans_and_animals",
    },
    "TRIMETHOPRIM": {
        "who_mia_category": "HIA",
        "who_mia_class": "Sulfonamides, dihydrofolate reductase inhibitors and combinations",
        "who_mapping_scope": "both_humans_and_animals",
    },
}
