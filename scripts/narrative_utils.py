"""
Shared narrative feature extraction utilities.

Provides extract_narrative_features() and NAV_FEATURES for reuse across
05_stratified_models_ads_l2.py and 06_narrative_features.py.

Design constraint: NO outcome words (injured, towed, airbag, etc.) are used
as features — only pre-crash context and crash dynamics.
"""
from __future__ import annotations

import re

import pandas as pd


NAV_FEATURES = [
    "nav_av_stopped",
    "nav_av_moving",
    "nav_av_turning",
    "nav_av_changing_lanes",
    "nav_av_reversing",
    "nav_av_struck_other",
    "nav_other_struck_av",
    "nav_rear_approach",
    "nav_at_intersection",
    "nav_on_highway",
    "nav_in_parking_lot",
    "nav_left_turn",
    "nav_lane_change",
    "nav_vulnerable_user",
    "nav_emergency_vehicle",
    "nav_large_vehicle",
    "nav_av_disengaged",
    "nav_hazard_lights",
    "nav_double_parked",
    "nav_speed_mentioned",
    "nav_minor_damage_lang",
]

NAV_DESCRIPTIONS = {
    "nav_av_stopped":        "AV was stopped/parked/stationary at time of impact",
    "nav_av_moving":         "AV was traveling/proceeding at time of impact",
    "nav_av_turning":        "AV was making a left/right/U-turn",
    "nav_av_changing_lanes": "AV was changing or merging lanes",
    "nav_av_reversing":      "AV was reversing/backing up",
    "nav_av_struck_other":   "AV initiated contact (AV struck the other party)",
    "nav_other_struck_av":   "Other party initiated contact (struck the AV)",
    "nav_rear_approach":     "Other party approached from behind / rear-end scenario",
    "nav_at_intersection":   "Crash occurred at intersection, red light, or stop sign",
    "nav_on_highway":        "Crash on freeway, highway, or interstate",
    "nav_in_parking_lot":    "Crash in parking lot or garage",
    "nav_left_turn":         "Left turn was involved in the crash sequence",
    "nav_lane_change":       "Lane change or merge was involved",
    "nav_vulnerable_user":   "Pedestrian, cyclist, or other vulnerable road user involved",
    "nav_emergency_vehicle": "Emergency or police vehicle involved",
    "nav_large_vehicle":     "Bus, semi-truck, or heavy vehicle involved",
    "nav_av_disengaged":     "Operator disengaged / took manual control near crash",
    "nav_hazard_lights":     "AV had hazard/emergency lights on (unusual situation flag)",
    "nav_double_parked":     "Double-parked or lane-blocking obstruction in the scene",
    "nav_speed_mentioned":   "A specific speed in mph was mentioned in the narrative",
    "nav_minor_damage_lang": '"Minor damage" or "no visible damage" language present',
}

_HEADER_PAT = re.compile(
    r"(?:Pursuant to|Under|Filed under|Submitted pursuant to|In accordance with)"
    r".*?(?:Standing General Order|SGO).*?(?:\.\s+|\n)",
    re.IGNORECASE | re.DOTALL,
)
_WAYMO_SUPPLEMENT = re.compile(
    r"Waymo may supplement.*?(?:\.\s+|\n)",
    re.IGNORECASE | re.DOTALL,
)


def clean_narrative(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = _HEADER_PAT.sub("", text)
    text = _WAYMO_SUPPLEMENT.sub("", text)
    text = re.sub(r"\[XXX\]", " ", text)
    text = re.sub(r"\[REDACTED[^\]]*\]", " ", text, flags=re.IGNORECASE)
    return text.strip()


def is_redacted(text: str) -> bool:
    if pd.isna(text):
        return True
    return bool(re.search(r"REDACTED|CBI|CONFIDENTIAL", str(text), re.IGNORECASE))


def extract_narrative_features(raw_text: str) -> dict:
    """Return binary scenario flags from a single narrative; all-zero if redacted."""
    flags = {f: 0 for f in NAV_FEATURES}
    flags["nav_has_narrative"] = 0

    if is_redacted(raw_text):
        return flags

    text = clean_narrative(raw_text)
    if len(text) < 20:
        return flags

    flags["nav_has_narrative"] = 1
    t = text.lower()

    flags["nav_av_stopped"] = int(bool(re.search(
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle)\b.*?"
        r"\b(stopped|stationary|parked|came to a stop|slowed to a stop|remained stopped|at rest)\b"
        r"|\b(stopped|stationary|parked|came to a stop|slowed to a stop|remained stopped)\b.*?"
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle)\b",
        t, re.DOTALL
    )))

    flags["nav_av_moving"] = int(bool(re.search(
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle)\b.*?"
        r"\b(traveling|proceeding|moving|driving|was in motion|drove)\b"
        r"|\b(traveling|proceeding|moving|driving)\b.*?"
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle)\b",
        t, re.DOTALL
    )))

    flags["nav_av_turning"] = int(bool(re.search(
        r"\b(left turn|right turn|u-turn|uturn|turning left|turning right|initiating a.{0,10}turn|executing.{0,15}turn)\b",
        t
    )))

    flags["nav_av_changing_lanes"] = int(bool(re.search(
        r"\b(chang(ing|ed) lanes?|lane change|merg(ing|ed)|chang(ing|ed) into)\b",
        t
    )))

    flags["nav_av_reversing"] = int(bool(re.search(
        r"\b(revers(ing|ed)|backing up|backed up|in reverse)\b", t
    )))

    flags["nav_av_struck_other"] = int(bool(re.search(
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle)\b.{0,80}"
        r"\b(struck|hit|made contact with|collided with|ran into)\b",
        t, re.DOTALL
    )))

    flags["nav_other_struck_av"] = int(bool(re.search(
        r"\b(struck|hit|made contact with|collided with|ran into|rear.ended|sideswiped)\b.{0,80}"
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle|stationary waymo|stopped waymo)\b"
        r"|\b(other.{0,15}(vehicle|party|car|truck|bus|cyclist|pedestrian))\b.{0,80}"
        r"\b(struck|hit|made contact with|collided with)\b",
        t, re.DOTALL
    )))

    flags["nav_rear_approach"] = int(bool(re.search(
        r"\b(approached.{0,30}from behind|rear.end(ed)?|from the rear|"
        r"approach(ed|ing) from behind|rear of the|behind the (waymo|av|subject)|"
        r"collided with the rear)\b",
        t
    )))

    flags["nav_at_intersection"] = int(bool(re.search(
        r"\b(intersection|red light|stop (sign|light)|traffic signal|yield sign|"
        r"crosswalk|at the light|at a light)\b",
        t
    )))

    flags["nav_on_highway"] = int(bool(re.search(
        r"\b(freeway|highway|interstate|i-\d+|us-\d+|expressway|on-ramp|off-ramp)\b",
        t
    )))

    flags["nav_in_parking_lot"] = int(bool(re.search(
        r"\b(parking lot|parking garage|parking structure|parking area)\b", t
    )))

    flags["nav_left_turn"] = int(bool(re.search(
        r"\b(left turn|turning left|turn(ing)? left|left-turn lane|dedicated.*left)\b", t
    )))

    flags["nav_lane_change"] = int(bool(re.search(
        r"\b(lane change|chang(ing|ed) lanes?|merg(ing|ed) (into|from)|"
        r"mov(ing|ed) (into|from).{0,20}lane)\b",
        t
    )))

    flags["nav_vulnerable_user"] = int(bool(re.search(
        r"\b(pedestrian|cyclist|bicyclist|bicycle|e.?bike|scooter|"
        r"person on foot|moped|motorcyclist)\b",
        t
    )))

    flags["nav_emergency_vehicle"] = int(bool(re.search(
        r"\b(police|fire truck|firetruck|ambulance|emergency vehicle|"
        r"first responder|law enforcement|patrol car)\b",
        t
    )))

    flags["nav_large_vehicle"] = int(bool(re.search(
        r"\b(bus|semi.?truck|semi|tractor.?trailer|delivery truck|"
        r"garbage truck|box truck|heavy truck|18.?wheeler)\b",
        t
    )))

    flags["nav_av_disengaged"] = int(bool(re.search(
        r"\b(disengaged|took (manual )?control|operator took over|"
        r"manual override|disengagement|switched to manual)\b",
        t
    )))

    flags["nav_hazard_lights"] = int(bool(re.search(
        r"\b(hazard lights?|emergency lights?|flashers|hazard flashers?)\b", t
    )))

    flags["nav_double_parked"] = int(bool(re.search(
        r"\b(double.?parked|blocking (the|a) lane|stopped in (the|a) (travel )?lane|"
        r"obstructing (the|a) lane)\b",
        t
    )))

    flags["nav_speed_mentioned"] = int(bool(re.search(
        r"\b\d+\s*mph\b|\bspeed of \d+\b|\btraveling at \d+\b", t
    )))

    flags["nav_minor_damage_lang"] = int(bool(re.search(
        r"\b(minor damage|minimal damage|no visible damage|cosmetic damage|"
        r"no damage|undamaged|slight damage)\b",
        t
    )))

    return flags


def attach_narrative_flags(df: pd.DataFrame, narrative_col: str = "Narrative") -> pd.DataFrame:
    """Add narrative flag columns to df in-place and return it."""
    flag_records = df[narrative_col].apply(extract_narrative_features)
    flag_df = pd.DataFrame(list(flag_records), index=df.index)
    return pd.concat([df, flag_df], axis=1)
