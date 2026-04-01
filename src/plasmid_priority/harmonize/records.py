"""Record-level harmonization for canonical PLSDB tables."""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import pandas as pd

from plasmid_priority.utils.dataframe import read_tsv


COUNTRY_ALIAS_GROUPS: dict[str, set[str]] = {
    "Afghanistan": {"AFGHANISTAN"},
    "Albania": {"ALBANIA"},
    "Algeria": {"ALGERIA"},
    "Andorra": {"ANDORRA"},
    "Angola": {"ANGOLA"},
    "Antigua and Barbuda": {"ANTIGUA AND BARBUDA"},
    "Argentina": {"ARGENTINA"},
    "Armenia": {"ARMENIA"},
    "Aruba": {"ARUBA"},
    "Australia": {"AUSTRALIA"},
    "Austria": {"AUSTRIA"},
    "Azerbaijan": {"AZERBAIJAN"},
    "Bahamas": {"BAHAMAS", "THE BAHAMAS"},
    "Bahrain": {"BAHRAIN"},
    "Bangladesh": {"BANGLADESH"},
    "Barbados": {"BARBADOS"},
    "Belarus": {"BELARUS"},
    "Belgium": {"BELGIUM"},
    "Belize": {"BELIZE"},
    "Benin": {"BENIN"},
    "Bermuda": {"BERMUDA"},
    "Bhutan": {"BHUTAN"},
    "Bolivia": {"BOLIVIA", "BOLIVIA PLURINATIONAL STATE OF"},
    "Bosnia and Herzegovina": {"BOSNIA AND HERZEGOVINA"},
    "Botswana": {"BOTSWANA"},
    "Brazil": {"BRAZIL"},
    "Brunei": {"BRUNEI", "BRUNEI DARUSSALAM"},
    "Bulgaria": {"BULGARIA"},
    "Burkina Faso": {"BURKINA FASO"},
    "Burundi": {"BURUNDI"},
    "Cambodia": {"CAMBODIA"},
    "Cameroon": {"CAMEROON"},
    "Canada": {"CANADA"},
    "Cape Verde": {"CAPE VERDE", "CABO VERDE"},
    "Cayman Islands": {"CAYMAN ISLANDS"},
    "Central African Republic": {"CENTRAL AFRICAN REPUBLIC"},
    "Chad": {"CHAD"},
    "Chile": {"CHILE"},
    "China": {"CHINA", "PEOPLES REPUBLIC OF CHINA", "PEOPLE S REPUBLIC OF CHINA", "PR CHINA", "P R CHINA"},
    "Colombia": {"COLOMBIA"},
    "Comoros": {"COMOROS"},
    "Costa Rica": {"COSTA RICA"},
    "Croatia": {"CROATIA"},
    "Cuba": {"CUBA"},
    "Curacao": {"CURACAO", "CURAÇAO"},
    "Cyprus": {"CYPRUS"},
    "Czech Republic": {"CZECH REPUBLIC", "CZECHIA"},
    "Democratic Republic of the Congo": {
        "DEMOCRATIC REPUBLIC OF THE CONGO",
        "DRC",
        "DR CONGO",
        "D R CONGO",
    },
    "Denmark": {"DENMARK"},
    "Djibouti": {"DJIBOUTI"},
    "Dominica": {"DOMINICA"},
    "Dominican Republic": {"DOMINICAN REPUBLIC"},
    "Ecuador": {"ECUADOR"},
    "Egypt": {"EGYPT"},
    "El Salvador": {"EL SALVADOR"},
    "Equatorial Guinea": {"EQUATORIAL GUINEA"},
    "Eritrea": {"ERITREA"},
    "Estonia": {"ESTONIA"},
    "Eswatini": {"ESWATINI", "SWAZILAND"},
    "Ethiopia": {"ETHIOPIA"},
    "Faroe Islands": {"FAROE ISLANDS"},
    "Fiji": {"FIJI"},
    "Finland": {"FINLAND"},
    "France": {"FRANCE"},
    "French Guiana": {"FRENCH GUIANA"},
    "French Polynesia": {"FRENCH POLYNESIA"},
    "Gabon": {"GABON"},
    "Gambia": {"GAMBIA", "THE GAMBIA"},
    "Georgia": {"GEORGIA"},
    "Germany": {"GERMANY"},
    "Ghana": {"GHANA"},
    "Gibraltar": {"GIBRALTAR"},
    "Greece": {"GREECE"},
    "Greenland": {"GREENLAND"},
    "Grenada": {"GRENADA"},
    "Guadeloupe": {"GUADELOUPE"},
    "Guam": {"GUAM"},
    "Guatemala": {"GUATEMALA"},
    "Guernsey": {"GUERNSEY"},
    "Guinea": {"GUINEA"},
    "Guinea-Bissau": {"GUINEA BISSAU"},
    "Guyana": {"GUYANA"},
    "Haiti": {"HAITI"},
    "Honduras": {"HONDURAS"},
    "Hong Kong": {"HONG KONG", "HONG KONG SAR"},
    "Hungary": {"HUNGARY"},
    "Iceland": {"ICELAND"},
    "India": {"INDIA"},
    "Indonesia": {"INDONESIA"},
    "Iran": {"IRAN", "IRAN ISLAMIC REPUBLIC OF"},
    "Iraq": {"IRAQ"},
    "Ireland": {"IRELAND", "REPUBLIC OF IRELAND"},
    "Isle of Man": {"ISLE OF MAN"},
    "Israel": {"ISRAEL"},
    "Italy": {"ITALY"},
    "Ivory Coast": {"IVORY COAST", "COTE D IVOIRE", "COTE DIVOIRE"},
    "Jamaica": {"JAMAICA"},
    "Japan": {"JAPAN"},
    "Jersey": {"JERSEY"},
    "Jordan": {"JORDAN"},
    "Kazakhstan": {"KAZAKHSTAN"},
    "Kenya": {"KENYA"},
    "Kosovo": {"KOSOVO"},
    "Kuwait": {"KUWAIT"},
    "Kyrgyzstan": {"KYRGYZSTAN"},
    "Laos": {"LAOS", "LAO PDR", "LAO PEOPLE S DEMOCRATIC REPUBLIC"},
    "Latvia": {"LATVIA"},
    "Lebanon": {"LEBANON"},
    "Lesotho": {"LESOTHO"},
    "Liberia": {"LIBERIA"},
    "Libya": {"LIBYA"},
    "Liechtenstein": {"LIECHTENSTEIN"},
    "Lithuania": {"LITHUANIA"},
    "Luxembourg": {"LUXEMBOURG"},
    "Macau": {"MACAU", "MACAO"},
    "Madagascar": {"MADAGASCAR"},
    "Malawi": {"MALAWI"},
    "Malaysia": {"MALAYSIA"},
    "Maldives": {"MALDIVES"},
    "Mali": {"MALI"},
    "Malta": {"MALTA"},
    "Martinique": {"MARTINIQUE"},
    "Mauritania": {"MAURITANIA"},
    "Mauritius": {"MAURITIUS"},
    "Mayotte": {"MAYOTTE"},
    "Mexico": {"MEXICO"},
    "Moldova": {"MOLDOVA", "REPUBLIC OF MOLDOVA"},
    "Monaco": {"MONACO"},
    "Mongolia": {"MONGOLIA"},
    "Montenegro": {"MONTENEGRO"},
    "Morocco": {"MOROCCO"},
    "Mozambique": {"MOZAMBIQUE"},
    "Myanmar": {"MYANMAR", "BURMA"},
    "Namibia": {"NAMIBIA"},
    "Nepal": {"NEPAL"},
    "Netherlands": {"NETHERLANDS", "THE NETHERLANDS"},
    "New Caledonia": {"NEW CALEDONIA"},
    "New Zealand": {"NEW ZEALAND"},
    "Nicaragua": {"NICARAGUA"},
    "Niger": {"NIGER"},
    "Nigeria": {"NIGERIA"},
    "North Korea": {"NORTH KOREA", "DPRK", "D P R K", "DEMOCRATIC PEOPLE S REPUBLIC OF KOREA"},
    "North Macedonia": {"NORTH MACEDONIA", "MACEDONIA", "REPUBLIC OF MACEDONIA"},
    "Norway": {"NORWAY"},
    "Oman": {"OMAN"},
    "Pakistan": {"PAKISTAN"},
    "Palestine": {"PALESTINE", "STATE OF PALESTINE", "WEST BANK", "GAZA"},
    "Panama": {"PANAMA"},
    "Papua New Guinea": {"PAPUA NEW GUINEA"},
    "Paraguay": {"PARAGUAY"},
    "Peru": {"PERU"},
    "Philippines": {"PHILIPPINES", "THE PHILIPPINES"},
    "Poland": {"POLAND"},
    "Portugal": {"PORTUGAL"},
    "Puerto Rico": {"PUERTO RICO"},
    "Qatar": {"QATAR"},
    "Republic of the Congo": {"REPUBLIC OF THE CONGO", "CONGO BRAZZAVILLE"},
    "Reunion": {"REUNION", "RÉUNION"},
    "Romania": {"ROMANIA"},
    "Russia": {"RUSSIA", "RUSSIAN FEDERATION"},
    "Rwanda": {"RWANDA"},
    "Saint Kitts and Nevis": {"SAINT KITTS AND NEVIS"},
    "Saint Lucia": {"SAINT LUCIA"},
    "Saint Vincent and the Grenadines": {"SAINT VINCENT AND THE GRENADINES"},
    "Samoa": {"SAMOA"},
    "San Marino": {"SAN MARINO"},
    "Sao Tome and Principe": {"SAO TOME AND PRINCIPE"},
    "Saudi Arabia": {"SAUDI ARABIA"},
    "Senegal": {"SENEGAL"},
    "Serbia": {"SERBIA"},
    "Seychelles": {"SEYCHELLES"},
    "Sierra Leone": {"SIERRA LEONE"},
    "Singapore": {"SINGAPORE"},
    "Slovakia": {"SLOVAKIA"},
    "Slovenia": {"SLOVENIA"},
    "Somalia": {"SOMALIA"},
    "South Africa": {"SOUTH AFRICA"},
    "South Korea": {"SOUTH KOREA", "KOREA", "KOREA REPUBLIC OF", "REPUBLIC OF KOREA"},
    "South Sudan": {"SOUTH SUDAN"},
    "Spain": {"SPAIN"},
    "Sri Lanka": {"SRI LANKA"},
    "Sudan": {"SUDAN"},
    "Suriname": {"SURINAME"},
    "Sweden": {"SWEDEN"},
    "Switzerland": {"SWITZERLAND"},
    "Syria": {"SYRIA", "SYRIAN ARAB REPUBLIC"},
    "Taiwan": {"TAIWAN", "TAIWAN ROC", "TAIWAN R O C", "REPUBLIC OF CHINA"},
    "Tajikistan": {"TAJIKISTAN"},
    "Tanzania": {"TANZANIA", "UNITED REPUBLIC OF TANZANIA"},
    "Thailand": {"THAILAND"},
    "Togo": {"TOGO"},
    "Trinidad and Tobago": {"TRINIDAD AND TOBAGO"},
    "Tunisia": {"TUNISIA"},
    "Turkey": {"TURKEY", "TURKIYE", "TÜRKIYE", "TÜRKIYE", "REPUBLIC OF TURKEY"},
    "Turkmenistan": {"TURKMENISTAN"},
    "Uganda": {"UGANDA"},
    "Ukraine": {"UKRAINE"},
    "United Arab Emirates": {"UNITED ARAB EMIRATES", "UAE", "U A E"},
    "UK": {
        "UNITED KINGDOM",
        "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND",
        "UK",
        "U K",
        "GREAT BRITAIN",
        "ENGLAND",
        "SCOTLAND",
        "WALES",
        "NORTHERN IRELAND",
    },
    "USA": {
        "UNITED STATES",
        "UNITED STATES OF AMERICA",
        "USA",
        "U S A",
        "US",
        "U S",
        "U.S.A",
        "U.S.",
        "U S A",
    },
    "Uruguay": {"URUGUAY"},
    "Uzbekistan": {"UZBEKISTAN"},
    "Venezuela": {"VENEZUELA", "VENEZUELA BOLIVARIAN REPUBLIC OF"},
    "Vietnam": {"VIETNAM", "VIET NAM"},
    "Virgin Islands, U.S.": {"US VIRGIN ISLANDS", "U S VIRGIN ISLANDS", "VIRGIN ISLANDS U S"},
    "Yemen": {"YEMEN"},
    "Zambia": {"ZAMBIA"},
    "Zimbabwe": {"ZIMBABWE"},
}

LOCATION_SEGMENT_SPLIT = re.compile(r"[,;:/|()\[\]]+")


def _normalize_location_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_country_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in COUNTRY_ALIAS_GROUPS.items():
        for alias in aliases | {canonical.upper()}:
            lookup[_normalize_location_key(alias)] = canonical
    return lookup


COUNTRY_LOOKUP = _build_country_lookup()
COUNTRY_MAX_TOKEN_LENGTH = max(len(alias.split()) for alias in COUNTRY_LOOKUP)


def _clean_marker_list(value: object) -> str:
    if pd.isna(value):
        return ""
    items = sorted({item.strip() for item in str(value).split(",") if item.strip()})
    return ",".join(items)


def _marker_count(value: object) -> int:
    cleaned = _clean_marker_list(value)
    return 0 if not cleaned else len(cleaned.split(","))


def normalize_country(value: object) -> str:
    """Extract a canonical country token from a biosample location field.

    The parser is intentionally conservative: it only returns a country when a
    curated country/territory alias is found somewhere in the location text.
    Free-text addresses, institutions, road names, and coordinates therefore
    resolve to the empty string instead of contaminating the country field.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""

    segments = [segment.strip() for segment in LOCATION_SEGMENT_SPLIT.split(text) if segment.strip()]
    if not segments:
        segments = [text]

    for segment in reversed(segments):
        normalized = _normalize_location_key(segment)
        if not normalized:
            continue
        if normalized in COUNTRY_LOOKUP:
            return COUNTRY_LOOKUP[normalized]

        tokens = normalized.split()
        max_length = min(COUNTRY_MAX_TOKEN_LENGTH, len(tokens))
        for length in range(max_length, 0, -1):
            for start in range(len(tokens) - length, -1, -1):
                candidate = " ".join(tokens[start : start + length])
                if candidate in COUNTRY_LOOKUP:
                    return COUNTRY_LOOKUP[candidate]
    return ""


def build_harmonized_plasmid_table(
    inventory_path: Path,
    typing_path: Path,
    biosample_path: Path,
) -> pd.DataFrame:
    """Merge inventory metadata, typing annotations, and biosample location fields."""
    canonical = read_tsv(inventory_path)

    typing = pd.read_csv(
        typing_path,
        usecols=[
            "NUCCORE_ACC",
            "gc",
            "size",
            "num_contigs",
            "rep_type(s)",
            "relaxase_type(s)",
            "mpf_type",
            "orit_type(s)",
            "predicted_mobility",
            "mash_neighbor_distance",
            "predicted_host_range_overall_rank",
            "predicted_host_range_overall_name",
            "reported_host_range_lit_rank",
            "reported_host_range_lit_name",
            "associated_pmid(s)",
            "primary_cluster_id",
            "secondary_cluster_id",
            "observed_host_range_ncbi_rank",
            "observed_host_range_ncbi_name",
            "PMLST_scheme",
            "PMLST_sequence_type",
            "PMLST_alleles",
        ],
    ).rename(columns={"NUCCORE_ACC": "sequence_accession"})

    biosample = pd.read_csv(
        biosample_path,
        usecols=[
            "BIOSAMPLE_UID",
            "LOCATION_name",
            "LOCATION_query",
            "BIOSAMPLE_title",
            "BIOSAMPLE_package",
            "BIOSAMPLE_pathogenicity",
            "ECOSYSTEM_tags",
            "DISEASE_tags",
        ],
    ).rename(columns={"BIOSAMPLE_UID": "biosample_uid"})

    harmonized = canonical.merge(typing, on="sequence_accession", how="left", validate="m:1")
    harmonized = harmonized.merge(biosample, on="biosample_uid", how="left", validate="m:1")

    harmonized["country"] = (
        harmonized["LOCATION_name"].fillna(harmonized["LOCATION_query"]).map(normalize_country)
    )
    harmonized["replicon_types"] = harmonized["rep_type(s)"].map(_clean_marker_list)
    harmonized["relaxase_types"] = harmonized["relaxase_type(s)"].map(_clean_marker_list)
    harmonized["orit_types"] = harmonized["orit_type(s)"].map(_clean_marker_list)
    harmonized["n_replicon_types"] = harmonized["replicon_types"].map(_marker_count)
    harmonized["n_relaxase_types"] = harmonized["relaxase_types"].map(_marker_count)
    harmonized["n_orit_types"] = harmonized["orit_types"].map(_marker_count)
    harmonized["primary_replicon"] = harmonized["replicon_types"].str.split(",").str[0].fillna("")
    harmonized["predicted_mobility"] = harmonized["predicted_mobility"].fillna("unknown").astype(str)
    harmonized["primary_cluster_id"] = harmonized["primary_cluster_id"].fillna("").astype(str)
    harmonized["mash_neighbor_distance"] = pd.to_numeric(
        harmonized.get("mash_neighbor_distance"),
        errors="coerce",
    ).fillna(0.0)
    harmonized["predicted_host_range_overall_rank"] = (
        harmonized.get("predicted_host_range_overall_rank", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    harmonized["predicted_host_range_overall_name"] = (
        harmonized.get("predicted_host_range_overall_name", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["reported_host_range_lit_rank"] = (
        harmonized.get("reported_host_range_lit_rank", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    harmonized["reported_host_range_lit_name"] = (
        harmonized.get("reported_host_range_lit_name", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["associated_pmid(s)"] = (
        harmonized.get("associated_pmid(s)", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["PMLST_scheme"] = (
        harmonized.get("PMLST_scheme", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["PMLST_sequence_type"] = (
        harmonized.get("PMLST_sequence_type", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["PMLST_alleles"] = (
        harmonized.get("PMLST_alleles", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["BIOSAMPLE_pathogenicity"] = (
        harmonized.get("BIOSAMPLE_pathogenicity", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["ECOSYSTEM_tags"] = (
        harmonized.get("ECOSYSTEM_tags", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["DISEASE_tags"] = (
        harmonized.get("DISEASE_tags", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["typing_gc"] = pd.to_numeric(harmonized.get("gc"), errors="coerce").fillna(0.0)
    harmonized["typing_size"] = pd.to_numeric(harmonized.get("size"), errors="coerce").fillna(0.0)
    harmonized["typing_num_contigs"] = pd.to_numeric(harmonized.get("num_contigs"), errors="coerce").fillna(0.0)

    harmonized["has_country"] = harmonized["country"].astype(str).str.len() > 0
    harmonized["has_relaxase"] = harmonized["n_relaxase_types"] > 0
    harmonized["has_mpf"] = harmonized["mpf_type"].fillna("").astype(str).str.len() > 0
    harmonized["has_orit"] = harmonized["n_orit_types"] > 0
    harmonized["is_mobilizable"] = harmonized["predicted_mobility"].isin(["mobilizable", "conjugative"])
    harmonized["is_conjugative"] = harmonized["predicted_mobility"].eq("conjugative")

    return harmonized
