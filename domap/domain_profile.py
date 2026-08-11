# -*- coding: utf-8 -*-
"""
FAA2CHART/domain_profile.py

Domain Profile configuration.

Framework code (MinerU calls, batch processing, JSON repair, retry, disk writing) is
decoupled from domain knowledge (what to extract, how to determine relevance, what schema to output).

Adding a new domain = adding a new DomainProfile instance, without modifying any framework code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class DomainProfile:
    """
    All declarative knowledge required for a domain.

    name:               Domain identifier, used for logging and default filenames.
    prompt_template:    Extraction prompt; must contain the {batch_text} placeholder.
    is_relevant:        Determines whether a text/table chunk belongs to this domain.
    clean_file_name:    Default filename for the cleaned intermediate output.
    final_file_name:    Default filename for the structured result.
    record_word:        Log label for a single record (e.g., "airway", "event").
    gazetteer_config:   Geocoding configuration. None means use default FAA NASR CSV.
                        dict format: {"type": "light_list_json", "json_path": "..."}
    analysis_strategy:  Spatial analysis strategy identifier, determines which analysis
                        step implementation the framework loads.
                        "airway_connectivity" → airway connectivity (linear, default)
                        "point_event"         → independent point events (point)
    """

    name: str
    prompt_template: str
    is_relevant: Callable[[str], bool]
    clean_file_name: str
    final_file_name: str
    record_word: str = "record"
    gazetteer_config: Optional[dict] = None
    analysis_strategy: str = "airway_connectivity"


# =====================================================================
# FAA Airway Domain (behaviour identical to pre-refactor)
# =====================================================================

FAA_PROMPT_TEMPLATE = r"""
Task: Extract airway route data from HTML tables or plain text in the input dictionary, output a pure JSON array. Do NOT output any explanations, comments, Markdown, or extra text.

Input:
{batch_text}

Input is a dict:
- Keys may be airway designators like V408
- Keys may also be prefixed titles like "95.6404 VOR FEDERAL AIRWAY V404" — extract V404 from these
- If key is none_table_* / none_text_*, extract airway designator from the value; if not extractable, use the key itself
- If key ends with "- CONTINUED", it indicates a continuation from the previous page for the same airway; merge with the previous airway after removing the suffix; if the last point of the previous segment equals the first point of the current segment, deduplicate when concatenating

Content parsing:
1. HTML tables
- Common columns: FROM, TO, MEA, MAA
- Ignore header/description rows like "AIRWAY SEGMENT", "CHANGEOVER POINTS", "FROM", "TO", "DISTANCE"
- If <td colspan="...">...V405...</td> appears, it indicates a new airway begins; subsequent content belongs to the new airway
- Column 3 extracts MEA, column 4 extracts MAA; keep only digits, e.g., *4700 → 4700
- If missing, fill "none"; if only 3 columns, MAA="none"

2. Plain text
- Each line is typically a segment, e.g.:
  UPNAR, OP WP COMIR, OP WP 5500
  Meaning: FROM=UPNAR, TO=COMIR, MEA=5500, MAA=none
- Multiple lines connect in order to form a complete airway
- If line ends without altitude, MEA="none", MAA="none"
- If a new airway designator appears in the text, split into a new airway

Waypoint structure:
{{
  "name": "point name",
  "region": "region code",
  "type": "point type",
  "position": ["none", "none"]
}}

Point parsing example:
GRAND TURK, TC VORTAC →
name=GRAND TURK, region=TC, type=VORTAC

Rules:
- Waypoints are output in order
- Consecutive duplicate points are kept only once
- If multiple airway designators appear in the same content block, they must be split
- If other new designators appear in a continuation page, also split into new airways and continue merging with their respective subsequent continuations

Output format:
[
  {{
    "airway_code": "V404",
    "MEA": "3000",
    "MAA": "60000",
    "airway_point": [
      {{
        "name": "GRAND TURK",
        "region": "TC",
        "type": "VORTAC",
        "position": ["none", "none"]
      }}
    ]
  }}
]

Hard requirements:
- Only output valid JSON array
- Fill "none" when MEA or MAA is missing
- When coordinates are missing, position is fixed as ["none","none"]
"""

_ROUTE_KEYWORDS_RE = re.compile(
    r"\b(FIX|WP|VOR\/DME|VORTAC|VOR|DME|NDB|MARINE NDB|FAN MARKER|TACAN|INTXN|WAYPOINT)\b",
    re.IGNORECASE,
)

_ALTITUDE_RE = re.compile(
    r"(\b\d{3,5}\b)|(\*\*\d{3,5})|(\*\d{3,5})|(-\s*(MOCA|MCA))",
    re.IGNORECASE,
)

_US_STATE_STYLE_RE = re.compile(
    r",\s*[A-Z]{2}\s+(FIX|WP|VOR\/DME|VORTAC|VOR|DME|NDB|TACAN|INTXN|WAYPOINT)\b"
)


def is_route_content(content: str) -> bool:
    """Determine whether a chunk of content is FAA airway content (logic identical to pre-refactor)."""
    if not content:
        return False

    text = content.strip()
    if not text:
        return False

    kw_hits = len(_ROUTE_KEYWORDS_RE.findall(text))
    if kw_hits < 1:
        return False

    alt_hit = bool(_ALTITUDE_RE.search(text))
    state_style_hit = bool(_US_STATE_STYLE_RE.search(text))

    return alt_hit or state_style_hit


FAA_AIRWAY_PROFILE = DomainProfile(
    name="faa_airway",
    prompt_template=FAA_PROMPT_TEMPLATE,
    is_relevant=is_route_content,
    clean_file_name="table_with_title-new3.json",
    final_file_name="parsed_routes_newnew_5.json",
    record_word="airway",
)


# =====================================================================
# USCG Maritime Notice Domain (for generalization validation)
# =====================================================================

USCG_LNM_PROMPT_TEMPLATE = r"""
Task: Extract aid-to-navigation change events from USCG Local Notice to Mariners text/tables in the input dictionary, output a pure JSON array. Do NOT output any explanations, comments, Markdown, or extra text.

Input:
{batch_text}

Input is a dict:
- Keys are section titles or waterway names, e.g., "SECTION II - DISCREPANCIES", "TENNESSEE RIVER"
- Values are the body text under that section
- If the value does not contain a waterway name, infer the waterway from the key

Content parsing:
- Each record describes a status change of an Aid to Navigation
- Common action words: DISCONTINUE / ESTABLISH / CHANGE / RELOCATE / DAMAGED / DESTROYED /
  MISSING / EXTINGUISHED / RECOVERED / REBUILT / TEMPORARY
- If a single body text contains multiple changes, split into multiple records

Field parsing:
- aid_name: Full name of the aid, e.g., "Holt Island Daybeacon"
- llnr: Light List Number, keep only digits, e.g., LLNR-3005 → 3005
- action: Action word, standardized to past participle uppercase, e.g., Discontinue → DISCONTINUED
- waterway: The waterway/river/harbor name
- mile_marker: River mile, e.g., mile 44.3 → 44.3; null if missing
- bank: LDB (Left Descending Bank) or RDB (Right Descending Bank); null if missing
- position: [latitude, longitude] in decimal degrees

Coordinate conversion rules:
- DMS format 35-59-36.145N/084-11-04.202W:
  degrees + minutes/60 + seconds/3600; South (S) and West (W) are negative
  Example → [35.993374, -84.184501]
- If already decimal, use directly
- When no coordinates, position = [null, null]

Output format:
[
  {{
    "aid_name": "Holt Island Daybeacon",
    "llnr": "3005",
    "action": "DISCONTINUED",
    "waterway": "Tennessee River",
    "mile_marker": "44.3",
    "bank": "RDB",
    "position": [35.993374, -84.184501]
  }}
]

Hard requirements:
- Only output valid JSON array
- Fill null for missing fields (do NOT use string "none")
- Do NOT fabricate facility names or coordinates
- Must process all sections in the input
"""

_MARITIME_ACTION_RE = re.compile(
    r"\b(DISCONTINUE[DS]?|ESTABLISH(?:ED|ING)?|CHANGE[DS]?|RELOCAT(?:E|ED|ING)?|"
    r"DAMAGED?|DESTROY(?:ED)?|MISSING|EXTINGUISH(?:ED)?|RECOVER(?:ED)?|"
    r"REBUILT|TEMPORARILY|RESET|WATCHING PROPERLY|"
    # USCG LNM common table abbreviations
    r"DMGD|LT EXT|STRUCT DMGD|TRUB|DBD DMGD|RELIGHTED|REDUCED INT|DAYMK MISSING|LT OUT|"
    r"DISCONTINUED|WATCHING|RELIT|OFF STA|WITHDRAWN|"
    r"DESTROYED|ADJUSTED|MOVED|REESTABLISHED|REINSTATED)\b",
    re.IGNORECASE,
)

_MARITIME_AID_RE = re.compile(
    r"\b(LLNR|DAYBEACON|DAYBOARD|DAYMARK|LIGHTED BUOY|BUOY|BEACON|LIGHT|"
    r"RANGE|AID TO NAVIGATION|ATON|FD\b|PA\b|"
    # HTML table LNM column names
    r"STATUS.*AID TYPE|NAME.*LLNR|AID TYPE|LIGHT LIST)\b",
    re.IGNORECASE,
)

_MARITIME_POSITION_RE = re.compile(
    r"\d{1,3}[-\u00b0]\d{1,2}[-'\u2032]\d{1,2}(?:\.\d+)?\s*[\"\u2033]?\s*[NS]",
    re.IGNORECASE,
)


def is_maritime_content(content: str) -> bool:
    """Determine whether a chunk of content is USCG aid-to-navigation change content.

    Isomorphic to FAA determination: first require domain noun hits, then require
    action words or coordinates, thereby excluding abbreviation tables, section
    descriptions, and other noise.
    """
    if not content:
        return False

    text = content.strip()
    if not text:
        return False

    aid_hits = len(_MARITIME_AID_RE.findall(text))
    if aid_hits < 1:
        return False

    action_hit = bool(_MARITIME_ACTION_RE.search(text))
    position_hit = bool(_MARITIME_POSITION_RE.search(text))

    return action_hit or position_hit


USCG_LNM_PROFILE = DomainProfile(
    name="uscg_lnm",
    prompt_template=USCG_LNM_PROMPT_TEMPLATE,
    is_relevant=is_maritime_content,
    clean_file_name="lnm_with_title.json",
    final_file_name="parsed_maritime_events.json",
    record_word="aids-to-navigation event",
    gazetteer_config={
        "type": "light_list_json",
        # json_path is no longer hardcoded; it is passed by the user via natural language
        # task description. The geocode sub-agent must extract the gazetteer path from the
        # user's task when calling the coordinate completion tool.
    },
    analysis_strategy="point_event",
)


PROFILES = {
    FAA_AIRWAY_PROFILE.name: FAA_AIRWAY_PROFILE,
    USCG_LNM_PROFILE.name: USCG_LNM_PROFILE,
}
