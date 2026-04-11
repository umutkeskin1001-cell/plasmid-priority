"""Geographic helper utilities for backbone-level reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Import the canonical country source of truth from shared constants
from plasmid_priority.utils.country_constants import COUNTRY_ALIAS_GROUPS

MACRO_REGION_ORDER = [
    "Europe",
    "North America",
    "Latin America and Caribbean",
    "Africa",
    "Asia",
    "Middle East and West Asia",
    "Oceania",
]

# Additional macro-region mappings for countries in COUNTRY_ALIAS_GROUPS
# but not yet in COUNTRY_TO_MACRO_REGION
_ADDITIONAL_COUNTRY_TO_MACRO_REGION = {
    "Andorra": "Europe",
    "Angola": "Africa",
    "Antigua and Barbuda": "Latin America and Caribbean",
    "Aruba": "Latin America and Caribbean",
    "Azerbaijan": "Middle East and West Asia",
    "Belize": "Latin America and Caribbean",
    "Benin": "Africa",
    "Bermuda": "North America",
    "Bhutan": "Asia",
    "Bosnia and Herzegovina": "Europe",
    "Brunei": "Asia",
    "Cape Verde": "Africa",
    "Cayman Islands": "Latin America and Caribbean",
    "Cyprus": "Europe",
    "Djibouti": "Africa",
    "Dominica": "Latin America and Caribbean",
    "Equatorial Guinea": "Africa",
    "Eritrea": "Africa",
    "Eswatini": "Africa",
    "Fiji": "Oceania",
    "French Guiana": "Latin America and Caribbean",
    "French Polynesia": "Oceania",
    "Gibraltar": "Europe",
    "Grenada": "Latin America and Caribbean",
    "Guam": "Oceania",
    "Guinea": "Africa",
    "Guyana": "Latin America and Caribbean",
    "Isle of Man": "Europe",
    "Ivory Coast": "Africa",
    "Kosovo": "Europe",
    "Latvia": "Europe",
    "Lesotho": "Africa",
    "Liberia": "Africa",
    "Liechtenstein": "Europe",
    "Macau": "Asia",
    "Maldives": "Asia",
    "Malta": "Europe",
    "Mauritania": "Africa",
    "Moldova": "Europe",
    "Monaco": "Europe",
    "Mozambique": "Africa",
    "North Korea": "Asia",
    "North Macedonia": "Europe",
    "Paraguay": "Latin America and Caribbean",
    "Saint Kitts and Nevis": "Latin America and Caribbean",
    "Saint Lucia": "Latin America and Caribbean",
    "Saint Vincent and the Grenadines": "Latin America and Caribbean",
    "San Marino": "Europe",
    "Sao Tome and Principe": "Africa",
    "Seychelles": "Africa",
    "Tajikistan": "Asia",
    "Togo": "Africa",
    "Turkmenistan": "Asia",
    "Uzbekistan": "Asia",
    "Virgin Islands, U.S.": "Latin America and Caribbean",
}

COUNTRY_TO_MACRO_REGION = {
    "Afghanistan": "Asia",
    "Albania": "Europe",
    "Algeria": "Africa",
    "Argentina": "Latin America and Caribbean",
    "Armenia": "Middle East and West Asia",
    "Australia": "Oceania",
    "Austria": "Europe",
    "Bahamas": "Latin America and Caribbean",
    "Bahrain": "Middle East and West Asia",
    "Bangladesh": "Asia",
    "Barbados": "Latin America and Caribbean",
    "Belarus": "Europe",
    "Belgium": "Europe",
    "Bolivia": "Latin America and Caribbean",
    "Botswana": "Africa",
    "Brazil": "Latin America and Caribbean",
    "Bulgaria": "Europe",
    "Burkina Faso": "Africa",
    "Burundi": "Africa",
    "Cambodia": "Asia",
    "Cameroon": "Africa",
    "Canada": "North America",
    "Central African Republic": "Africa",
    "Chad": "Africa",
    "Chile": "Latin America and Caribbean",
    "China": "Asia",
    "Colombia": "Latin America and Caribbean",
    "Comoros": "Africa",
    "Costa Rica": "Latin America and Caribbean",
    "Croatia": "Europe",
    "Cuba": "Latin America and Caribbean",
    "Curacao": "Latin America and Caribbean",
    "Czech Republic": "Europe",
    "Democratic Republic of the Congo": "Africa",
    "Denmark": "Europe",
    "Dominican Republic": "Latin America and Caribbean",
    "Ecuador": "Latin America and Caribbean",
    "Egypt": "Africa",
    "El Salvador": "Latin America and Caribbean",
    "Estonia": "Europe",
    "Ethiopia": "Africa",
    "Faroe Islands": "Europe",
    "Finland": "Europe",
    "France": "Europe",
    "Gabon": "Africa",
    "Gambia": "Africa",
    "Georgia": "Middle East and West Asia",
    "Germany": "Europe",
    "Ghana": "Africa",
    "Greece": "Europe",
    "Greenland": "North America",
    "Guadeloupe": "Latin America and Caribbean",
    "Guatemala": "Latin America and Caribbean",
    "Guernsey": "Europe",
    "Guinea-Bissau": "Africa",
    "Haiti": "Latin America and Caribbean",
    "Honduras": "Latin America and Caribbean",
    "Hong Kong": "Asia",
    "Hungary": "Europe",
    "Iceland": "Europe",
    "India": "Asia",
    "Indonesia": "Asia",
    "Iran": "Middle East and West Asia",
    "Iraq": "Middle East and West Asia",
    "Ireland": "Europe",
    "Israel": "Middle East and West Asia",
    "Italy": "Europe",
    "Jamaica": "Latin America and Caribbean",
    "Japan": "Asia",
    "Jersey": "Europe",
    "Jordan": "Middle East and West Asia",
    "Kazakhstan": "Asia",
    "Kenya": "Africa",
    "Kuwait": "Middle East and West Asia",
    "Kyrgyzstan": "Asia",
    "Laos": "Asia",
    "Lebanon": "Middle East and West Asia",
    "Libya": "Africa",
    "Lithuania": "Europe",
    "Luxembourg": "Europe",
    "Madagascar": "Africa",
    "Malawi": "Africa",
    "Malaysia": "Asia",
    "Mali": "Africa",
    "Martinique": "Latin America and Caribbean",
    "Mauritius": "Africa",
    "Mayotte": "Africa",
    "Mexico": "Latin America and Caribbean",
    "Mongolia": "Asia",
    "Montenegro": "Europe",
    "Morocco": "Africa",
    "Mozambique": "Africa",
    "Myanmar": "Asia",
    "Namibia": "Africa",
    "Nepal": "Asia",
    "Netherlands": "Europe",
    "New Caledonia": "Oceania",
    "New Zealand": "Oceania",
    "Nicaragua": "Latin America and Caribbean",
    "Niger": "Africa",
    "Nigeria": "Africa",
    "Norway": "Europe",
    "Oman": "Middle East and West Asia",
    "Pakistan": "Asia",
    "Palestine": "Middle East and West Asia",
    "Panama": "Latin America and Caribbean",
    "Papua New Guinea": "Oceania",
    "Peru": "Latin America and Caribbean",
    "Philippines": "Asia",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Puerto Rico": "Latin America and Caribbean",
    "Qatar": "Middle East and West Asia",
    "Republic of the Congo": "Africa",
    "Reunion": "Africa",
    "Romania": "Europe",
    "Russia": "Europe",
    "Rwanda": "Africa",
    "Samoa": "Oceania",
    "Saudi Arabia": "Middle East and West Asia",
    "Senegal": "Africa",
    "Serbia": "Europe",
    "Seychelles": "Africa",
    "Sierra Leone": "Africa",
    "Singapore": "Asia",
    "Slovakia": "Europe",
    "Slovenia": "Europe",
    "Somalia": "Africa",
    "South Africa": "Africa",
    "South Korea": "Asia",
    "South Sudan": "Africa",
    "Spain": "Europe",
    "Sri Lanka": "Asia",
    "Sudan": "Africa",
    "Suriname": "Latin America and Caribbean",
    "Sweden": "Europe",
    "Switzerland": "Europe",
    "Syria": "Middle East and West Asia",
    "Taiwan": "Asia",
    "Tanzania": "Africa",
    "Thailand": "Asia",
    "Trinidad and Tobago": "Latin America and Caribbean",
    "Tunisia": "Africa",
    "Turkey": "Middle East and West Asia",
    "UK": "Europe",
    "USA": "North America",
    "Uganda": "Africa",
    "Ukraine": "Europe",
    "United Arab Emirates": "Middle East and West Asia",
    "Uruguay": "Latin America and Caribbean",
    "Venezuela": "Latin America and Caribbean",
    "Vietnam": "Asia",
    "Yemen": "Middle East and West Asia",
    "Zambia": "Africa",
    "Zimbabwe": "Africa",
    # Merge in the additional mappings to ensure full coverage
    **_ADDITIONAL_COUNTRY_TO_MACRO_REGION,
}


def country_to_macro_region(country: object) -> str:
    """Map a canonical country token to a coarse macro-region."""
    text = str(country or "").strip()
    if not text:
        return ""
    return COUNTRY_TO_MACRO_REGION.get(text, "")


def validate_country_macro_region_coverage() -> list[str]:
    """Return list of canonical countries missing from COUNTRY_TO_MACRO_REGION.

    This validates that all countries defined in COUNTRY_ALIAS_GROUPS
    have a corresponding macro-region mapping. An empty list indicates
    full coverage.
    """
    canonical_countries = set(COUNTRY_ALIAS_GROUPS.keys())
    mapped_countries = set(COUNTRY_TO_MACRO_REGION.keys())
    return sorted(canonical_countries - mapped_countries)


def dominant_macro_region_table(
    records: pd.DataFrame,
    *,
    split_year: int = 2015,
    country_column: str = "country",
    output_column: str = "dominant_region_train",
) -> pd.DataFrame:
    """Return the dominant mapped macro-region for each backbone in training rows."""
    training = records.loc[
        pd.to_numeric(records["resolved_year"], errors="coerce").fillna(0).astype(int) <= split_year
    ].copy()
    if training.empty:
        return pd.DataFrame(columns=["backbone_id", output_column])

    training[output_column] = training[country_column].map(country_to_macro_region)
    training[output_column] = training[output_column].fillna("").astype(str).str.strip()
    training = training.loc[training[output_column].ne(""), ["backbone_id", output_column]].copy()
    if training.empty:
        return pd.DataFrame(columns=["backbone_id", output_column])
    counts = (
        training.groupby(["backbone_id", output_column], as_index=False)
        .size()
        .sort_values(
            ["backbone_id", "size", output_column], ascending=[True, False, True], kind="mergesort"
        )
        .drop_duplicates("backbone_id", keep="first")[["backbone_id", output_column]]
        .reset_index(drop=True)
    )
    return counts


def build_country_quality_summary(
    records: pd.DataFrame,
    *,
    split_year: int = 2015,
) -> pd.DataFrame:
    """Summarize country-field completeness and macro-region mappability."""
    if records.empty:
        return pd.DataFrame()
    working = records.copy()
    years = pd.to_numeric(working["resolved_year"], errors="coerce").fillna(0).astype(int)
    working["country_clean"] = working["country"].fillna("").astype(str).str.strip()
    working["macro_region"] = working["country_clean"].map(country_to_macro_region)
    working["period"] = pd.Series(
        np.nan,
        index=working.index,
        dtype="object",
    )
    working.loc[years <= split_year, "period"] = "training"
    working.loc[years > split_year, "period"] = "later"

    rows = []
    for period_name, frame in [("all_rows", working)] + [
        (str(period), subset.copy())
        for period, subset in working.groupby("period", dropna=True, sort=False)
    ]:
        n_rows = int(len(frame))
        with_country = frame["country_clean"].ne("")
        mapped_region = frame["macro_region"].ne("")
        backbones_with_country = int(frame.loc[with_country, "backbone_id"].astype(str).nunique())
        total_backbones = int(frame["backbone_id"].astype(str).nunique())
        rows.append(
            {
                "period": period_name,
                "n_rows": n_rows,
                "n_rows_with_country": int(with_country.sum()),
                "country_non_null_fraction": float(with_country.mean()) if n_rows else 0.0,
                "n_unique_countries": int(frame.loc[with_country, "country_clean"].nunique()),
                "n_rows_with_macro_region": int(mapped_region.sum()),
                "macro_region_mapped_fraction": float(mapped_region.mean()) if n_rows else 0.0,
                "n_backbones_with_any_country": backbones_with_country,
                "n_backbones_without_country": max(total_backbones - backbones_with_country, 0),
            }
        )
    return pd.DataFrame(rows)
