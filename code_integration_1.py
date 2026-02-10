import pandas as pd
import os
import re
import numpy as np
from collections import defaultdict, Counter
import json
import folium

# =========================
# Config
# =========================
GTFS_DIR = "gtfs_bus_only"

TRIPS_PATH = os.path.join(GTFS_DIR, "trips.txt")
STOP_TIMES_PATH = os.path.join(GTFS_DIR, "stop_times.txt")
STOPS_PATH = os.path.join(GTFS_DIR, "stops.txt")
ROUTES_PATH = os.path.join(GTFS_DIR, "routes.txt")

BLOCK_SUMMARY_PATH = "block_analysis_final.csv"

SERVICE_MAP = {1: "MF", 2: "Sat", 3: "Sun"}

# Candidate highlights (optional)
CANDIDATE_TXT = "on_route_charger_location.txt"  # if you still want green highlights


# =========================
# Helpers
# =========================
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


def classify_bus_len(asset_class: str) -> str:
    """
    Map asset_class into:
      28ft, 40ft, 60ft, 44ft_dd (double decker), other
    Adjust patterns if your naming differs.
    """
    if not isinstance(asset_class, str):
        return "other"

    s = asset_class.strip().lower()

    # 60-foot articulated
    if "60" in s or "60ft" in s or "artic" in s or "articulated" in s:
        return "60ft"

    # 44-foot double decker
    if "44" in s or "double" in s or "dd" in s or "decker" in s:
        return "44ft_dd"

    # 40-foot standard
    if "40" in s or "40ft" in s:
        return "40ft"

    # 28-foot community / shuttle
    if "28" in s or "28ft" in s:
        return "28ft"

    return "other"



def union_sets(series):
    combined = set()
    for x in series:
        if pd.isna(x):
            continue
        if isinstance(x, (set, list, tuple, np.ndarray)):
            combined |= set(x)
        else:
            combined.add(x)
    return combined


def safe_list(x):
    if isinstance(x, (set, list, tuple, np.ndarray)):
        return list(x)
    return [x]


# =========================
# Load base files once
# =========================
trips_all = pd.read_csv(TRIPS_PATH, dtype={"trip_id": str, "route_id": str, "block_id": str})
stop_times_all = pd.read_csv(STOP_TIMES_PATH, dtype={"trip_id": str, "stop_id": str})
stops = pd.read_csv(STOPS_PATH, dtype={"stop_id": str, "stop_code": str})
routes = pd.read_csv(ROUTES_PATH, dtype={"route_id": str, "route_short_name": str})

block_success_summary_all = pd.read_csv(BLOCK_SUMMARY_PATH)

# Normalize block_id types everywhere (important)
trips_all["block_id"] = trips_all["block_id"].apply(norm_block_id)
block_success_summary_all["block_id"] = block_success_summary_all["block_id"].apply(norm_block_id)

# Ensure service_id exists in both sources
if "service_id" not in trips_all.columns:
    raise ValueError("trips.txt must contain 'service_id' column for MF/Sat/Sun split.")
if "service_id" not in block_success_summary_all.columns:
    raise ValueError("block_analysis_final.csv must contain 'service_id' column for MF/Sat/Sun split.")

trips_all["service_id"] = pd.to_numeric(trips_all["service_id"], errors="coerce")
block_success_summary_all["service_id"] = pd.to_numeric(block_success_summary_all["service_id"], errors="coerce")

# Stops: keep stop_code and lat/lon
stops_key = (
    stops.drop_duplicates(subset=["stop_code"])
         [["stop_code", "stop_name", "stop_lat", "stop_lon", "stop_id"]]
         .copy()
)

# Merge stop_code onto stop_times
stop_times_all = stop_times_all.merge(
    stops[["stop_id", "stop_code"]],
    on="stop_id",
    how="left"
)

# Optional highlight set
highlight_set = set()
if os.path.exists(CANDIDATE_TXT):
    curr_candidate = pd.read_csv(CANDIDATE_TXT)
    curr_candidate["Location_new"] = curr_candidate["Location"].replace({
        "Commercial-Broadway Station": "N Grandview Hwy",
        "Knight Street-Marine Drive": "Northbound Knight St Bridge Offramp"
    })
    highlight_set = set(curr_candidate["Location_new"].astype(str))


# =========================
# Main loop: MF / Sat / Sun
# =========================
for sid, label in SERVICE_MAP.items():
    print(f"\n=== Building map for service_id={sid} ({label}) ===")

    # ---- Filter block summary + trips by service_id
    block_success_summary = block_success_summary_all[block_success_summary_all["service_id"] == sid].copy()
    valid_block_ids = set(block_success_summary["block_id"].dropna().unique())

    trips = trips_all[(trips_all["service_id"] == sid) & (trips_all["block_id"].isin(valid_block_ids))].copy()
    if trips.empty:
        print(f"[WARN] No trips found for service_id={sid}. Skipping.")
        continue

    valid_trip_ids = set(trips["trip_id"].unique())
    stop_times = stop_times_all[stop_times_all["trip_id"].isin(valid_trip_ids)].copy()

    # ---- Add route_short_name
    trips_with_route = trips.merge(
        routes[["route_id", "route_short_name"]],
        on="route_id",
        how="left"
    )

    trip_to_block = trips_with_route.set_index("trip_id")["block_id"].to_dict()
    trip_to_route_short = trips_with_route.set_index("trip_id")["route_short_name"].to_dict()

    # ---- block_id -> asset_class / depot (from your block_analysis_final.csv)
    # Prefer asset_class_new if present
    asset_col = "asset_class_new" if "asset_class_new" in block_success_summary.columns else "asset_class"
    block_to_asset_class = block_success_summary.set_index("block_id")[asset_col].to_dict()
    block_to_depot = block_success_summary.set_index("block_id")["depot_code"].to_dict()

    # ---- First/last stops by trip (by stop_code)
    stop_times_sorted = stop_times.sort_values(["trip_id", "stop_sequence"])
    first_stops = stop_times_sorted.groupby("trip_id", as_index=False).first()[["trip_id", "stop_code"]]
    last_stops  = stop_times_sorted.groupby("trip_id", as_index=False).last()[["trip_id", "stop_code"]]

    start_dict = defaultdict(set)
    for _, row in first_stops.dropna(subset=["stop_code"]).iterrows():
        start_dict[str(row["stop_code"])].add(row["trip_id"])

    end_dict = defaultdict(set)
    for _, row in last_stops.dropna(subset=["stop_code"]).iterrows():
        end_dict[str(row["stop_code"])].add(row["trip_id"])

    all_stop_codes = sorted(set(start_dict.keys()) | set(end_dict.keys()))

    # ---- Base per-stop_code table
    rows = []
    for code in all_stop_codes:
        start_set = start_dict.get(code, set())
        end_set = end_dict.get(code, set())
        all_trips = start_set | end_set
        rows.append({
            "stop_code": code,
            "start_trip_ids": start_set,
            "end_trip_ids": end_set,
            "all_trip_ids": all_trips,
            "n_trips": len(all_trips),
        })

    trip_sets_df = pd.DataFrame(rows)

    # Merge stop metadata (lat/lon/name) by stop_code
    result_df = trip_sets_df.merge(
        stops_key.drop_duplicates(subset=["stop_code"])[["stop_code", "stop_name", "stop_lat", "stop_lon"]],
        on="stop_code",
        how="left"
    )

    # ---- Converters
    def trips_to_block_set(trip_ids):
        blocks = set()
        for t in safe_list(trip_ids):
            b = trip_to_block.get(t)
            if b is not None and str(b).strip() != "":
                blocks.add(str(b))
        return blocks

    def trips_to_route_short_set(trip_ids):
        routes_short = set()
        for t in safe_list(trip_ids):
            r = trip_to_route_short.get(t)
            if isinstance(r, str) and r.strip() != "":
                routes_short.add(r)
        return routes_short

    def blocks_to_bus_type_counter(block_ids):
        """
        Returns dict counting UNIQUE blocks by bus type.
        Supported keys:
        28ft, 40ft, 60ft, 44ft_dd, other
        """
        ctr = Counter()

        for b in safe_list(block_ids):
            ac = block_to_asset_class.get(b)
            ctr[classify_bus_len(ac)] += 1

        # Ensure all expected keys exist (even if zero)
        return {
            "28ft": int(ctr.get("28ft", 0)),
            "40ft": int(ctr.get("40ft", 0)),
            "60ft": int(ctr.get("60ft", 0)),
            "44ft_dd": int(ctr.get("44ft_dd", 0)),
            "other": int(ctr.get("other", 0)),
        }


    def blocks_to_depot_counter(block_ids):
        """
        Returns dict: { depot_code: n_blocks } counting UNIQUE blocks.
        """
        ctr = Counter()
        for b in safe_list(block_ids):
            d = block_to_depot.get(b)
            if isinstance(d, str) and d.strip() != "":
                ctr[d.strip()] += 1
            else:
                ctr["UNKNOWN"] += 1
        return {k: int(v) for k, v in sorted(ctr.items(), key=lambda x: (-x[1], x[0]))}

    # ---- Normalize stop_name_simple (same logic as you)
    result_df["stop_name_simple"] = (
        result_df["stop_name"]
        .astype(str)
        .str.replace(r"\s*@.*$", "", regex=True)
        .str.strip()
    )

    # ---- Group by stop_name_simple (terminal cluster)
    grouped_df = (
        result_df
        .groupby("stop_name_simple", as_index=False)
        .agg({
            "stop_code":      union_sets,
            "start_trip_ids": union_sets,
            "end_trip_ids":   union_sets,
            "all_trip_ids":   union_sets,
            "stop_name":      "first",
            "stop_lat":       "first",
            "stop_lon":       "first",
        })
    )

    grouped_df["num_trip_total"]  = grouped_df["all_trip_ids"].apply(len)
    grouped_df["num_trip_starts"] = grouped_df["start_trip_ids"].apply(len)
    grouped_df["num_trip_ends"]   = grouped_df["end_trip_ids"].apply(len)

    grouped_df["stop_code"]      = grouped_df["stop_code"].apply(lambda s: sorted(map(str, s)))
    grouped_df["start_trip_ids"] = grouped_df["start_trip_ids"].apply(lambda s: sorted(map(str, s)))
    grouped_df["end_trip_ids"]   = grouped_df["end_trip_ids"].apply(lambda s: sorted(map(str, s)))
    grouped_df["all_trip_ids"]   = grouped_df["all_trip_ids"].apply(lambda s: sorted(map(str, s)))

    grouped_df["block_ids"] = grouped_df["all_trip_ids"].apply(trips_to_block_set)
    grouped_df["num_unique_blocks"] = grouped_df["block_ids"].apply(lambda s: len(set(s)))
    grouped_df["route_short_names"] = grouped_df["all_trip_ids"].apply(trips_to_route_short_set)

    # NEW popup dicts requested:
    grouped_df["unique_blocks_by_bus_type"] = grouped_df["block_ids"].apply(blocks_to_bus_type_counter)
    grouped_df["unique_blocks_by_depot"] = grouped_df["block_ids"].apply(blocks_to_depot_counter)

    # Make lists for display
    grouped_df["block_ids"] = grouped_df["block_ids"].apply(lambda s: sorted(list(set(s))))
    grouped_df["route_short_names"] = grouped_df["route_short_names"].apply(lambda s: sorted(list(set(s))))

    # ---- Build folium map
    center_lat = grouped_df["stop_lat"].mean()
    center_lon = grouped_df["stop_lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    max_activity = grouped_df["num_trip_total"].max() if len(grouped_df) else 0

    for _, row in grouped_df.iterrows():
        activity = row["num_trip_total"]
        radius = 5 + 25 * (activity / max_activity) if max_activity > 0 else 5

        stop_codes_str = ", ".join(sorted(map(str, row["stop_code"])))

        route_short_names = row.get("route_short_names", [])
        routes_str = ", ".join(map(str, route_short_names)) if isinstance(route_short_names, (list, set, tuple)) else str(route_short_names)

        # NEW dict fields
        bus_type_dict = row.get("unique_blocks_by_bus_type", {})
        depot_dict = row.get("unique_blocks_by_depot", {})

        # Marker color: highlight terminals if in candidate list
        if str(row["stop_name_simple"]) in highlight_set:
            color = fill_color = "green"
        else:
            color = fill_color = "blue"

        folium.CircleMarker(
            location=[row["stop_lat"], row["stop_lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.4,
            weight=1,
            popup=folium.Popup(
                f"<b>{row['stop_name_simple']}</b><br>"
                f"Service: {label} (service_id={sid})<br>"
                f"Unique Blocks: {row['num_unique_blocks']}<br>"
                f"Unique Routes: {routes_str}<br>"
                f"Unique Blocks by Bus Type: {json.dumps(bus_type_dict, ensure_ascii=False)}<br>"
                f"Unique Blocks by Depot: {json.dumps(depot_dict, ensure_ascii=False)}<br>"
                f"Total Trips: {row['num_trip_total']}<br>"
                # f"Trip Starts: {row['num_trip_starts']}<br>"
                # f"Trip Ends: {row['num_trip_ends']}<br>"
                f"Stop Codes: {stop_codes_str}<br>",
                max_width=420
            )
        ).add_to(m)

    out_html = f"terminal_stations_grouped_map_{label}_service_id_{sid}.html"
    m.save(out_html)
    print(f"[OK] Map saved to {out_html}")
