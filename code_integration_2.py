"""
Terminal minute occupancy (dropoff / recovery / pickup) from GTFS blocks,
using stops.stop_code (NOT stop_id) to identify candidate terminal stops.

Key fix implemented:
✅ Dominant-duration minute assignment:
   For each (terminal, minute, block_id) the bus is assigned to exactly ONE status
   (dropoff OR recovery OR pickup) based on the most seconds overlapped in that minute.
   This removes the "duplication" you observed at minute boundaries.

Other requirements implemented:
- Run separately for service_id in {1,2,3} (1=MF,2=Sat,3=Sun) and save separate files
- No timestamp column; add day_offset and is_next_day (e.g., 25:10 is next day)
- Drop rows where total == 0
- Add description_json ONLY to terminal_minute_counts_by_terminal_* table
  as a JSON string grouped by status and metadata counts.
- Metadata (line_group, block_number, asset_class, depot_code) comes from block_analysis_final.csv
  joined via normalized block_id (handles int vs '21003.0' etc.)
- prev_route_short_name / next_route_short_name comes from GTFS routes.txt (via trips.route_id)

Inputs:
- GTFS folder with stops.txt, stop_times.txt, trips.txt, routes.txt
- Candidate terminal excel: df_to_save_candidate.xlsx (stop_name_simple + stop_code list)
- Block meta csv: block_analysis_final.csv

Outputs per service_id:
- terminal_minute_counts_{MF|Sat|Sun}_service_id_{sid}.csv
- terminal_minute_counts_by_terminal_{MF|Sat|Sun}_service_id_{sid}.csv   (includes description_json)
- terminal_intervals_{MF|Sat|Sun}_service_id_{sid}.csv
"""

import os
import re
import math
import json
from collections import defaultdict, Counter
import pandas as pd
import numpy as np


# =======================
# User inputs / paths
# =======================
CANDIDATE_XLSX = "df_to_save_candidate.xlsx"
BLOCK_META_CSV = "block_analysis_final.csv"  
SERVICE_MAP = {1: "MF", 2: "Sat", 3: "Sun"}


# =======================
# Parameters (assumptions)
# =======================
DROP_SEC = 90
PICK_SEC = 90
DEADHEAD_DIST_M = 500
ARRIVE_BEFORE_DEP_SEC = int(6.5 * 60)  # 390
RECOVERY_SEC = 5 * 60                  # 300


# =======================
# Helpers
# =======================
def parse_gtfs_time_to_sec(t: str) -> int:
    """GTFS time can exceed 24:00:00 (e.g., 25:13:00). Convert to seconds since service day start."""
    if pd.isna(t):
        return np.nan
    hh, mm, ss = t.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def sec_to_hhmm(sec: int) -> str:
    """Preserves 25:xx style display (no modulo 24h)."""
    hh = sec // 3600
    mm = (sec % 3600) // 60
    return f"{hh:02d}:{mm:02d}"


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in meters."""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return float(2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def _norm_stop_code(x) -> str:
    """Normalize stop_code coming from Excel and GTFS to a comparable string."""
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return str(int(x)) if float(x).is_integer() else str(x)
    s = str(x).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def norm_block_id(x):
    """Normalize block_id to comparable string: handles int, float-like '21003.0', whitespace."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        return s[:-2]
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def safe_split_same_terminal(arr_s: int, dep_s: int):
    """
    Split [arr_s, dep_s) into dropoff / recovery / pickup robustly even if dwell is short.
    Returns list of (status, start, end).
    """
    if dep_s <= arr_s:
        return []

    dwell = dep_s - arr_s
    drop = min(DROP_SEC, dwell / 2.0)
    pick = min(PICK_SEC, max(0.0, dwell - drop))
    drop = int(math.floor(drop))
    pick = int(math.floor(pick))

    rec_start = arr_s + drop
    rec_end = dep_s - pick

    out = []
    if drop > 0:
        out.append(("dropoff", arr_s, arr_s + drop))
    if rec_end > rec_start:
        out.append(("recovery", rec_start, rec_end))
    if pick > 0:
        out.append(("pickup", dep_s - pick, dep_s))
    return out


def add_interval_minute_diff(diff: np.ndarray, start_s: int, end_s: int):
    """
    Adds +1 to each minute overlapped by [start_s, end_s).
    Uses a difference array for O(1) range updates.
    """
    if end_s <= start_s:
        return
    m0 = start_s // 60
    m1 = (end_s - 1) // 60  # inclusive last minute touched
    if m0 < 0:
        m0 = 0
    if m1 < 0:
        return
    if m0 >= len(diff):
        return
    m1 = min(m1, len(diff) - 2)  # keep room for m1+1
    diff[m0] += 1
    diff[m1 + 1] -= 1


# def _detect_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str:
#     """Pick the first existing column name from candidates."""
#     for c in candidates:
#         if c in df.columns:
#             return c
#     if required:
#         raise ValueError(f"Missing required column. Tried: {candidates}. Available: {list(df.columns)[:50]} ...")
#     return ""


# =======================
# Dominant-duration minute assignment
# =======================
STATUS_PRIORITY = {"pickup": 3, "dropoff": 2, "recovery": 1}  # tie-break only

def build_minute_assignment(intervals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert second-level intervals to minute-level assignment:
      (terminal, minute_index, block_id) -> exactly ONE status
    using dominant overlap seconds in that minute.
    """
    if intervals_df.empty:
        return pd.DataFrame(columns=["terminal", "minute_index", "time_sec", "block_id", "status", "dominant_sec"])

    # (terminal, minute, block_id, status) -> overlap_seconds
    acc = defaultdict(int)

    for _, r in intervals_df.iterrows():
        term = str(r["terminal"])
        block_id = str(r["block_id"])
        status = r["status"]
        s = int(r["start_sec"])
        e = int(r["end_sec"])
        if e <= s:
            continue

        m0 = s // 60
        m1 = (e - 1) // 60

        for m in range(m0, m1 + 1):
            ms = m * 60
            me = (m + 1) * 60
            overlap = min(e, me) - max(s, ms)
            if overlap > 0:
                acc[(term, m, block_id, status)] += overlap

    # choose dominant status per (terminal, minute, block_id)
    by_key = defaultdict(list)
    for (term, m, block_id, status), sec in acc.items():
        by_key[(term, m, block_id)].append((status, sec))

    out = []
    for (term, m, block_id), items in by_key.items():
        items_sorted = sorted(
            items,
            key=lambda x: (x[1], STATUS_PRIORITY.get(x[0], 0)),
            reverse=True,
        )
        best_status, best_sec = items_sorted[0]
        out.append({
            "terminal": term,
            "minute_index": int(m),
            "time_sec": int(m) * 60,
            "block_id": block_id,
            "status": best_status,
            "dominant_sec": int(best_sec),
        })

    return pd.DataFrame(out)


def build_per_terminal_minute_from_assignment(assign_df: pd.DataFrame, service_id: int, service_label: str) -> pd.DataFrame:
    """
    Aggregate minute assignment into terminal_minute_counts_by_terminal table.
    Ensures no duplication across statuses per bus per minute.
    """
    if assign_df.empty:
        return pd.DataFrame()

    a = assign_df.copy()
    a["day_offset"] = (a["time_sec"] // 86400).astype(int)
    a["is_next_day"] = (a["day_offset"] >= 1).astype(int)
    a["time_hhmm"] = a["time_sec"].apply(sec_to_hhmm)

    # each row is (terminal, minute, block_id) with a single status
    g = a.groupby(["terminal", "time_sec", "time_hhmm", "day_offset", "is_next_day"], as_index=False)

    rows = []
    for _, df in g:
        term = df["terminal"].iloc[0]
        time_sec = int(df["time_sec"].iloc[0])
        time_hhmm = df["time_hhmm"].iloc[0]
        day_offset = int(df["day_offset"].iloc[0])
        is_next_day = int(df["is_next_day"].iloc[0])

        total = int(df["block_id"].nunique())
        dropoff = int((df["status"] == "dropoff").sum())
        recovery = int((df["status"] == "recovery").sum())
        pickup = int((df["status"] == "pickup").sum())

        rows.append({
            "service_id": service_id,
            "service_label": service_label,
            "terminal": term,
            "time_sec": time_sec,
            "time_hhmm": time_hhmm,
            "day_offset": day_offset,
            "is_next_day": is_next_day,
            "total": total,
            "dropoff": dropoff,
            "recovery": recovery,
            "pickup": pickup,
        })

    per_terminal_df = pd.DataFrame(rows)
    per_terminal_df = per_terminal_df[per_terminal_df["total"] > 0].reset_index(drop=True)
    return per_terminal_df


def add_description_json_to_per_terminal_from_assignment(
    per_terminal_df: pd.DataFrame,
    assign_df: pd.DataFrame,
    bus_lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build description_json for per_terminal_df using the minute assignment table,
    so each bus contributes to exactly one status per minute.

    JSON groups by status, then by metadata tuple with counts n:
      line_group, block_number, asset_class, depot_code, prev_route, next_route
    """
    if per_terminal_df.empty:
        out = per_terminal_df.copy()
        out["description_json"] = []
        return out

    # Only keep keys that exist in per_terminal_df
    valid_keys = set(zip(per_terminal_df["terminal"].astype(str),
                         (per_terminal_df["time_sec"].values // 60).astype(int)))

    # Join assignment -> per-boundary metadata lookup
    a = assign_df.merge(bus_lookup_df, on=["terminal", "block_id", "time_sec"], how="left")

    key_status_counter = defaultdict(lambda: defaultdict(Counter))

    for _, r in a.iterrows():
        term = str(r["terminal"])
        m = int(r["time_sec"]) // 60
        if (term, m) not in valid_keys:
            continue

        st = r["status"]
        meta_key = (
            str(r.get("line_group", "")),
            str(r.get("block_number", "")),
            str(r.get("asset_class", "")),
            str(r.get("depot_code", "")),
            str(r.get("prev_route_short_name", "")),
            str(r.get("next_route_short_name", "")),
        )
        key_status_counter[(term, m)][st][meta_key] += 1

    def norm(x: str):
        return None if x in ("nan", "None", "") else x

    def key_to_dict(k, n):
        return {
            "line_group": norm(k[0]),
            "block_number": norm(k[1]),
            "asset_class": norm(k[2]),
            "depot_code": norm(k[3]),
            "prev_route": norm(k[4]),
            "next_route": norm(k[5]),
            "n": int(n),
        }

    json_list = []
    for _, row in per_terminal_df.iterrows():
        term = str(row["terminal"])
        m = int(row["time_sec"]) // 60
        payload = {"dropoff": [], "recovery": [], "pickup": []}
        ctrs = key_status_counter.get((term, m), {})

        for st in ("dropoff", "recovery", "pickup"):
            ctr = ctrs.get(st, Counter())
            payload[st] = [key_to_dict(k, n) for k, n in ctr.items()]

        json_list.append(json.dumps(payload, ensure_ascii=False))

    out = per_terminal_df.copy()
    out["description_json"] = json_list
    return out


# =======================
# Core computation (one service_id)
# =======================
def build_terminal_minute_counts_for_service(
    gtfs_dir: str,
    service_id: int,
    candidate_xlsx: str = CANDIDATE_XLSX,
    block_meta_csv: str = BLOCK_META_CSV,
    deadhead_dist_m: int = DEADHEAD_DIST_M,
):
    # ---- Load candidate terminals (stop_code based) ----
    cand = pd.read_excel(candidate_xlsx)

    stopcode_to_terminal = {}
    terminals = []

    for _, r in cand.iterrows():
        term = str(r["stop_name_simple"])
        terminals.append(term)
        codes = r["stop_code"]

        if isinstance(codes, str):
            try:
                codes = eval(codes)
            except Exception:
                codes = [codes]
        if not isinstance(codes, (list, tuple, np.ndarray)):
            codes = [codes]

        for c in codes:
            sc = _norm_stop_code(c)
            if sc:
                stopcode_to_terminal[sc] = term

    candidate_stop_codes = set(stopcode_to_terminal.keys())
    candidate_terminal_set = set(terminals)

    # ---- Load GTFS ----
    stops = pd.read_csv(os.path.join(gtfs_dir, "stops.txt"), dtype={"stop_id": str, "stop_code": str})
    trips = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"), dtype={"trip_id": str, "block_id": str, "route_id": str})
    stop_times = pd.read_csv(os.path.join(gtfs_dir, "stop_times.txt"), dtype={"trip_id": str, "stop_id": str})
    routes = pd.read_csv(os.path.join(gtfs_dir, "routes.txt"), dtype={"route_id": str, "route_short_name": str})

    if "service_id" not in trips.columns:
        raise ValueError("trips.txt has no 'service_id' column. Please confirm your GTFS feed format.")

    trips["service_id"] = pd.to_numeric(trips["service_id"], errors="coerce")
    trips = trips[(trips["service_id"] == service_id)].dropna(subset=["block_id"]).copy()
    if trips.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # normalize trips block_id
    trips["block_id_norm"] = trips["block_id"].apply(norm_block_id)
    trips["block_id"] = trips["block_id_norm"]
    trips = trips.dropna(subset=["block_id"]).copy()

    # route short name
    routeid_to_short = routes.set_index("route_id")["route_short_name"].to_dict()
    trips["route_short_name"] = trips["route_id"].map(routeid_to_short)

    # stops mapping: stop_id -> stop_code + lat/lon
    if "stop_code" not in stops.columns:
        raise ValueError("stops.txt has no stop_code column. Confirm your GTFS feed.")

    stops["stop_code_norm"] = stops["stop_code"].apply(_norm_stop_code)
    stop_id_to_code = stops.set_index("stop_id")["stop_code_norm"].to_dict()
    stop_id_to_lat = stops.set_index("stop_id")["stop_lat"].to_dict()
    stop_id_to_lon = stops.set_index("stop_id")["stop_lon"].to_dict()

    # ---- Load block meta table and normalize block_id ----
    meta = pd.read_csv(block_meta_csv)
    if "service_id" not in meta.columns:
        raise ValueError("block meta csv must include service_id column.")

    meta["service_id"] = pd.to_numeric(meta["service_id"], errors="coerce")
    meta = meta[meta["service_id"] == service_id].copy()

    col_block_id = "block_id"
    col_line_group = "line_group"
    col_block_number = "block_number"
    col_depot = "depot_code"

    asset_col = "asset_class"

    meta["block_id_norm"] = meta[col_block_id].apply(norm_block_id)

    meta_std = pd.DataFrame({
        "block_id": meta["block_id_norm"].astype(str),
        "line_group": meta[col_line_group] if col_line_group else np.nan,
        "block_number": meta[col_block_number] if col_block_number else np.nan,
        "depot_code": meta[col_depot] if col_depot else np.nan,
        "asset_class": meta[asset_col] if asset_col else np.nan,
    }).drop_duplicates("block_id")

    # ---- Trip endpoints ----
    stop_times["arrival_sec"] = stop_times["arrival_time"].apply(parse_gtfs_time_to_sec)
    stop_times["departure_sec"] = stop_times["departure_time"].apply(parse_gtfs_time_to_sec)
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")

    st_sorted = stop_times.sort_values(["trip_id", "stop_sequence"])

    first = st_sorted.groupby("trip_id", as_index=False).first()[["trip_id", "stop_id", "departure_sec"]]
    first = first.rename(columns={"stop_id": "first_stop_id", "departure_sec": "trip_start_sec"})

    last = st_sorted.groupby("trip_id", as_index=False).last()[["trip_id", "stop_id", "arrival_sec"]]
    last = last.rename(columns={"stop_id": "last_stop_id", "arrival_sec": "trip_end_sec"})

    trip_ends = (
        first.merge(last, on="trip_id", how="inner")
             .merge(trips[["trip_id", "block_id", "route_short_name"]], on="trip_id", how="inner")
             .merge(meta_std, on="block_id", how="left")
    )

    trip_ends["first_stop_code"] = trip_ends["first_stop_id"].map(stop_id_to_code)
    trip_ends["last_stop_code"] = trip_ends["last_stop_id"].map(stop_id_to_code)
    trip_ends["first_lat"] = trip_ends["first_stop_id"].map(stop_id_to_lat)
    trip_ends["first_lon"] = trip_ends["first_stop_id"].map(stop_id_to_lon)
    trip_ends["last_lat"] = trip_ends["last_stop_id"].map(stop_id_to_lat)
    trip_ends["last_lon"] = trip_ends["last_stop_id"].map(stop_id_to_lon)

    trip_ends["first_in_candidates"] = trip_ends["first_stop_code"].isin(candidate_stop_codes)
    trip_ends["last_in_candidates"] = trip_ends["last_stop_code"].isin(candidate_stop_codes)
    trip_ends["first_terminal"] = trip_ends["first_stop_code"].map(stopcode_to_terminal)
    trip_ends["last_terminal"] = trip_ends["last_stop_code"].map(stopcode_to_terminal)

    trip_ends = trip_ends.sort_values(["block_id", "trip_start_sec"]).reset_index(drop=True)

    # ---- Build intervals ----
    intervals = []

    def add_interval(row, terminal, status, s, e, prev_route, next_route):
        if terminal not in candidate_terminal_set:
            return
        if e <= s:
            return
        intervals.append({
            "service_id": service_id,
            "block_id": str(row["block_id"]),
            "terminal": terminal,
            "status": status,
            "start_sec": int(s),
            "end_sec": int(e),
            "line_group": row.get("line_group", np.nan),
            "block_number": row.get("block_number", np.nan),
            "asset_class": row.get("asset_class", np.nan),
            "depot_code": row.get("depot_code", np.nan),
            "prev_route_short_name": prev_route,
            "next_route_short_name": next_route,
        })

    for block_id, g in trip_ends.groupby("block_id", sort=False):
        g = g.sort_values("trip_start_sec").reset_index(drop=True)

        # Case 3: first trip arrival (if first stop in candidates)
        if bool(g.loc[0, "first_in_candidates"]) and pd.notna(g.loc[0, "trip_start_sec"]):
            dep = int(g.loc[0, "trip_start_sec"])
            arr = max(0, dep - ARRIVE_BEFORE_DEP_SEC)

            rec_end = max(arr, dep - PICK_SEC)
            rec_start = max(arr, dep - PICK_SEC - RECOVERY_SEC)

            add_interval(g.loc[0], g.loc[0, "first_terminal"], "recovery", rec_start, rec_end,
                         prev_route=None, next_route=g.loc[0, "route_short_name"])
            add_interval(g.loc[0], g.loc[0, "first_terminal"], "pickup", dep - PICK_SEC, dep,
                         prev_route=None, next_route=g.loc[0, "route_short_name"])

        # Pairwise logic
        for i in range(len(g)):
            cur = g.loc[i]
            cur_end = cur["trip_end_sec"]
            cur_last_in = bool(cur["last_in_candidates"])
            cur_last_term = cur["last_terminal"]
            prev_route = cur["route_short_name"]

            # Case 4: last trip dropoff
            if i == len(g) - 1:
                if cur_last_in and pd.notna(cur_end):
                    arr = int(cur_end)
                    add_interval(cur, cur_last_term, "dropoff", arr, arr + DROP_SEC,
                                 prev_route=prev_route, next_route=None)
                break

            nxt = g.loc[i + 1]
            nxt_dep = nxt["trip_start_sec"]
            nxt_first_in = bool(nxt["first_in_candidates"])
            nxt_first_term = nxt["first_terminal"]
            next_route = nxt["route_short_name"]

            if pd.isna(cur_end) or pd.isna(nxt_dep):
                continue

            arr = int(cur_end)
            dep = int(nxt_dep)

            d_m = None
            try:
                d_m = haversine_m(cur["last_lat"], cur["last_lon"], nxt["first_lat"], nxt["first_lon"])
            except Exception:
                d_m = None

            same_terminal_candidate = (
                cur_last_in and nxt_first_in and
                (cur_last_term == nxt_first_term) and
                (d_m is not None and d_m <= deadhead_dist_m)
            )
            deadhead_like = (d_m is not None and d_m > deadhead_dist_m)

            # Case 1: same terminal
            if same_terminal_candidate:
                for status, s, e in safe_split_same_terminal(arr, dep):
                    add_interval(cur, cur_last_term, status, s, e,
                                 prev_route=prev_route, next_route=next_route)
                continue

            # Case 2: interline/deadhead
            if deadhead_like:
                if cur_last_in and nxt_first_in:
                    add_interval(cur, cur_last_term, "dropoff", arr, arr + DROP_SEC,
                                 prev_route=prev_route, next_route=next_route)

                    dest_dep = dep
                    dest_arr = max(0, dest_dep - ARRIVE_BEFORE_DEP_SEC)
                    rec_end = max(dest_arr, dest_dep - PICK_SEC)
                    rec_start = max(dest_arr, rec_end - RECOVERY_SEC)

                    add_interval(nxt, nxt_first_term, "recovery", rec_start, rec_end,
                                 prev_route=prev_route, next_route=next_route)
                    add_interval(nxt, nxt_first_term, "pickup", dest_dep - PICK_SEC, dest_dep,
                                 prev_route=prev_route, next_route=next_route)

                elif cur_last_in and (not nxt_first_in):
                    add_interval(cur, cur_last_term, "dropoff", arr, arr + DROP_SEC,
                                 prev_route=prev_route, next_route=next_route)

                elif (not cur_last_in) and nxt_first_in:
                    dest_dep = dep
                    dest_arr = max(0, dest_dep - ARRIVE_BEFORE_DEP_SEC)
                    rec_end = max(dest_arr, dest_dep - PICK_SEC)
                    rec_start = max(dest_arr, rec_end - RECOVERY_SEC)

                    add_interval(nxt, nxt_first_term, "recovery", rec_start, rec_end,
                                 prev_route=prev_route, next_route=next_route)
                    add_interval(nxt, nxt_first_term, "pickup", dest_dep - PICK_SEC, dest_dep,
                                 prev_route=prev_route, next_route=next_route)

                continue

            # everything else ignored

    intervals_df = pd.DataFrame(intervals)
    if intervals_df.empty:
        return intervals_df, pd.DataFrame(), pd.DataFrame()

    # ---- System-wide minute table (keep your original diff approach; duplication here is fine because it's total-level)
    max_end = int(intervals_df["end_sec"].max())
    nmin = (max_end // 60) + 3

    def make_series(status: str):
        diff = np.zeros(nmin, dtype=int)
        sub = intervals_df[intervals_df["status"] == status]
        for s, e in zip(sub["start_sec"].values, sub["end_sec"].values):
            add_interval_minute_diff(diff, int(s), int(e))
        return np.cumsum(diff)[:nmin - 1]

    dropoff = make_series("dropoff")
    recovery = make_series("recovery")
    pickup = make_series("pickup")
    total = dropoff + recovery + pickup

    minutes = np.arange(len(total), dtype=int)
    time_sec = (minutes * 60).astype(int)
    day_offset = (time_sec // 86400).astype(int)
    is_next_day = (day_offset >= 1).astype(int)

    minute_df = pd.DataFrame({
        "service_id": service_id,
        "service_label": SERVICE_MAP.get(service_id, str(service_id)),
        "time_sec": time_sec,
        "time_hhmm": [sec_to_hhmm(int(s)) for s in time_sec],
        "day_offset": day_offset,
        "is_next_day": is_next_day,
        "total_buses_in_candidates": total,
        "dropoff_buses": dropoff,
        "recovery_buses": recovery,
        "pickup_buses": pickup,
    })
    minute_df = minute_df[minute_df["total_buses_in_candidates"] > 0].reset_index(drop=True)

    # ---- Per-terminal minute table with dominant-duration (fixes duplication)
    assign_df = build_minute_assignment(intervals_df)
    per_terminal_df = build_per_terminal_minute_from_assignment(
        assign_df=assign_df,
        service_id=service_id,
        service_label=SERVICE_MAP.get(service_id, str(service_id)),
    )

    # ---- Build a per-minute lookup of metadata for JSON
    # We need prev/next routes at the minute level (which boundary is active in that minute).
    # Approach:
    # - For each (terminal, minute, block_id), pick the interval row that overlaps that minute with MAX seconds
    #   and inherit its metadata (line_group, block_number, asset_class, depot_code, prev_route, next_route).
    interval_acc = defaultdict(int)
    interval_meta = {}

    for _, r in intervals_df.iterrows():
        term = str(r["terminal"])
        block_id = str(r["block_id"])
        status = r["status"]
        s = int(r["start_sec"])
        e = int(r["end_sec"])
        if e <= s:
            continue
        m0 = s // 60
        m1 = (e - 1) // 60
        for m in range(m0, m1 + 1):
            ms = m * 60
            me = (m + 1) * 60
            overlap = min(e, me) - max(s, ms)
            if overlap <= 0:
                continue
            key = (term, m, block_id, status)
            interval_acc[key] += overlap
            # store metadata (same for the interval)
            interval_meta[key] = {
                "line_group": r.get("line_group", np.nan),
                "block_number": r.get("block_number", np.nan),
                "asset_class": r.get("asset_class", np.nan),
                "depot_code": r.get("depot_code", np.nan),
                "prev_route_short_name": r.get("prev_route_short_name", None),
                "next_route_short_name": r.get("next_route_short_name", None),
            }

    # Now for each assigned (terminal, minute, block_id), find the best matching interval key for that status
    bus_lookup_rows = []
    if not assign_df.empty:
        for _, arow in assign_df.iterrows():
            term = str(arow["terminal"])
            m = int(arow["minute_index"])
            time_s = int(arow["time_sec"])
            block_id = str(arow["block_id"])
            status = arow["status"]

            # pick the interval key for this (term,m,block,status) with max overlap
            best_k = None
            best_sec = -1
            k = (term, m, block_id, status)
            sec = interval_acc.get(k, 0)
            if sec > best_sec:
                best_sec = sec
                best_k = k

            meta_payload = interval_meta.get(best_k, {})
            bus_lookup_rows.append({
                "terminal": term,
                "block_id": block_id,
                "time_sec": time_s,
                **meta_payload
            })

    bus_lookup_df = pd.DataFrame(bus_lookup_rows)
    # Add JSON (based on assignment, so no double-count possible)
    per_terminal_df = add_description_json_to_per_terminal_from_assignment(
        per_terminal_df=per_terminal_df,
        assign_df=assign_df,
        bus_lookup_df=bus_lookup_df,
    )

    # Sanity check: should always hold with dominant-duration
    if not per_terminal_df.empty:
        ok = (per_terminal_df["total"] == (per_terminal_df["dropoff"] + per_terminal_df["recovery"] + per_terminal_df["pickup"]))
        if not bool(ok.all()):
            bad_n = int((~ok).sum())
            print(f"[WARN] {bad_n} rows violate total == dropoff+recovery+pickup (should be 0).")

    return intervals_df, minute_df, per_terminal_df


# =======================
# Runner: save per service_id
# =======================
def run_services_and_save(gtfs_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    for sid in [1, 2, 3]:
        intervals_df, minute_df, per_terminal_df = build_terminal_minute_counts_for_service(
            gtfs_dir=gtfs_dir,
            service_id=sid,
            candidate_xlsx=CANDIDATE_XLSX,
            block_meta_csv=BLOCK_META_CSV,
            deadhead_dist_m=DEADHEAD_DIST_M,
        )

        label = SERVICE_MAP.get(sid, f"service{sid}")

        intervals_path = os.path.join(out_dir, f"terminal_intervals_{label}_service_id_{sid}.csv")
        minute_path = os.path.join(out_dir, f"terminal_minute_counts_{label}_service_id_{sid}.csv")
        per_term_path = os.path.join(out_dir, f"terminal_minute_counts_by_terminal_{label}_service_id_{sid}.csv")

        intervals_df.to_csv(intervals_path, index=False)
        minute_df.to_csv(minute_path, index=False)
        per_terminal_df.to_csv(per_term_path, index=False)

        print(f"[OK] service_id={sid} ({label})")
        print(f"  minute:    {minute_path}")
        print(f"  per_term:  {per_term_path}")
        print(f"  intervals: {intervals_path}")




if __name__ == "__main__":
    gtfs_dir = r"gtfs_bus_only"          
    out_dir = r"output_term"    
    run_services_and_save(gtfs_dir, out_dir)