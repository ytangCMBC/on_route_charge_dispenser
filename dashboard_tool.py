import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

SERVICE_MAP = {1: "MF", 2: "Sat", 3: "Sun"}

BUSY_COLOR = {
    "Quiet": "#DCEAF7",
    "Moderate": "#9CC9F5",
    "Busy": "#4C9BE8",
    "Very Busy": "#1F5FA6",
}

def file_for_map(out_dir: str, sid: int) -> str:
    label = SERVICE_MAP.get(sid, str(sid))
    return os.path.join(out_dir, f"terminal_stations_grouped_map_{label}_service_id_{sid}.html")

def file_for_table(out_dir: str, sid: int) -> str:
    label = SERVICE_MAP.get(sid, str(sid))
    return os.path.join(out_dir, f"terminal_minute_counts_by_terminal_{label}_service_id_{sid}.csv")

def sec_to_hhmm(sec: int) -> str:
    hh = sec // 3600
    mm = (sec % 3600) // 60
    return f"{hh:02d}:{mm:02d}"

def assign_busy_level(total: int, q1: float, q2: float, q3: float) -> str:
    if total <= q1:
        return "Quiet"
    if total <= q2:
        return "Moderate"
    if total <= q3:
        return "Busy"
    return "Very Busy"

def build_time_slots(df: pd.DataFrame, slot_minutes: int) -> pd.DataFrame:
    d = df.copy()
    d["slot_index"] = (d["time_sec"] // (slot_minutes * 60)).astype(int)
    d["slot_start_sec"] = d["slot_index"] * slot_minutes * 60
    d["slot_label"] = d["slot_start_sec"].apply(sec_to_hhmm)

    agg = (
        d.groupby(["slot_index", "slot_start_sec", "slot_label"], as_index=False)
         .agg(
             total_peak=("total", "max"),
             dropoff=("dropoff", "sum"),
             recovery=("recovery", "sum"),
             pickup=("pickup", "sum"),
             n_minutes=("total", "size"),
         )
         .sort_values("slot_start_sec")
         .reset_index(drop=True)
    )
    return agg

def read_html(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_desc_json(s: str) -> dict:
    if not isinstance(s, str) or not s.strip():
        return {"dropoff": [], "recovery": [], "pickup": []}
    try:
        obj = json.loads(s)
        # ensure keys exist
        for k in ("dropoff", "recovery", "pickup"):
            obj.setdefault(k, [])
        return obj
    except Exception:
        return {"dropoff": [], "recovery": [], "pickup": []}

def explode_desc(desc_obj: dict, status: str) -> pd.DataFrame:
    items = desc_obj.get(status, []) if isinstance(desc_obj, dict) else []
    if not items:
        return pd.DataFrame(columns=["line_group","block_number","asset_class","depot_code","prev_route","next_route","n"])
    df = pd.DataFrame(items)
    # normalize columns order
    cols = ["line_group","block_number","asset_class","depot_code","prev_route","next_route","n"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    # cast n
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    return df

st.set_page_config(page_title="Terminal Occupancy Dashboard", layout="wide")
st.title("Terminal Occupancy Dashboard")

with st.sidebar:
    st.header("Data source")
    out_dir = st.text_input("output_term folder", value="output_term")

    st.divider()
    service_label = st.selectbox("Service day", options=["MF", "Sat", "Sun"], index=0)
    sid = {"MF": 1, "Sat": 2, "Sun": 3}[service_label]

    st.divider()
    slot_minutes = st.selectbox("Time slot size (minutes)", options=[5, 10, 15, 30, 60], index=2)

# Validate files exist
map_path = file_for_map(out_dir, sid)
table_path = file_for_table(out_dir, sid)

missing = []
if not os.path.exists(out_dir):
    missing.append(f"Folder not found: {out_dir}")
if not os.path.exists(map_path):
    missing.append(f"Map not found: {map_path}")
if not os.path.exists(table_path):
    missing.append(f"Minute table not found: {table_path}")

if missing:
    st.error("Missing required outputs:\n- " + "\n- ".join(missing))
    st.stop()

# -----------------------
# Map
# -----------------------
st.subheader(f"Terminal map — {service_label}")
st.components.v1.html(read_html(map_path), height=560, scrolling=True)

st.divider()

# -----------------------
# Minute breakdown
# -----------------------
st.subheader("Minute-level breakdown by terminal")
df = pd.read_csv(table_path)

required = {"terminal", "time_sec", "total", "dropoff", "recovery", "pickup"}
missing_cols = required - set(df.columns)
if missing_cols:
    st.error(f"Minute table is missing required columns: {sorted(missing_cols)}")
    st.stop()

# Ensure helper columns
if "time_hhmm" not in df.columns:
    df["time_hhmm"] = df["time_sec"].apply(sec_to_hhmm)
if "day_offset" not in df.columns:
    df["day_offset"] = (df["time_sec"] // 86400).astype(int)
if "is_next_day" not in df.columns:
    df["is_next_day"] = (df["day_offset"] >= 1).astype(int)

has_desc = "description_json" in df.columns

terminal_options = sorted(df["terminal"].dropna().unique().tolist())
selected_terminal = st.selectbox("Select terminal", options=terminal_options, index=0)

df_t = df[df["terminal"] == selected_terminal].copy()
df_t = df_t.sort_values("time_sec").reset_index(drop=True)

# Slots + busy colors
# Build slot table using current slot_minutes
slot_df = build_time_slots(df_t, slot_minutes=slot_minutes)

# Busy bins by quartiles of slot peak
q1, q2, q3 = slot_df["total_peak"].quantile([0.25, 0.50, 0.75]).tolist()
slot_df["busy_level"] = slot_df["total_peak"].apply(lambda x: assign_busy_level(int(x), q1, q2, q3))
slot_df["busy_color"] = slot_df["busy_level"].map(BUSY_COLOR)

# =========================
# Row 1: Busy timeline
# =========================
st.markdown(f"#### Busy timeline ({slot_minutes}-min slots)")

html = ["<div style='display:flex; flex-wrap:wrap; gap:6px;'>"]
for _, r in slot_df.iterrows():
    tooltip = f"{r['slot_label']} | peak={int(r['total_peak'])} | {r['busy_level']}"
    html.append(
        f"<div title='{tooltip}' "
        f"style='width:70px; height:26px; background:{r['busy_color']}; border-radius:8px; "
        f"border:1px solid rgba(0,0,0,0.12); display:flex; align-items:center; justify-content:center; "
        f"font-size:11px;'>"
        f"{r['slot_label']}</div>"
    )
html.append("</div>")
st.markdown("\n".join(html), unsafe_allow_html=True)

st.caption("Color indicates how busy the terminal is in each slot (based on peak total buses).")

# =========================
# Row 2: Slot-level stacked plot
# =========================
st.markdown(f"#### Status composition (bus count) — {slot_minutes}-min slots")

# Build slot index per minute
df_slot_bus = df_t.copy()
df_slot_bus["slot_index"] = (df_slot_bus["time_sec"] // (slot_minutes * 60)).astype(int)
df_slot_bus["slot_start_sec"] = df_slot_bus["slot_index"] * slot_minutes * 60
df_slot_bus["slot_label"] = df_slot_bus["slot_start_sec"].apply(sec_to_hhmm)

# Explode description_json to get block_id + status at minute level
rows = []
for _, r in df_slot_bus.iterrows():
    desc = parse_desc_json(r["description_json"])
    for status in ["dropoff", "recovery", "pickup"]:
        for item in desc.get(status, []):
            rows.append({
                "slot_label": r["slot_label"],
                "status": status,
                "block_number": item.get("block_number")
            })

df_slot_exploded = pd.DataFrame(rows)

# Count UNIQUE buses per slot per status
slot_bus_counts = (
    df_slot_exploded
    .dropna(subset=["block_number"])
    .groupby(["slot_label", "status"], as_index=False)
    .agg(bus_count=("block_number", "nunique"))
)

# Pivot to wide format for stacked plot
slot_bus_wide = (
    slot_bus_counts
    .pivot(index="slot_label", columns="status", values="bus_count")
    .fillna(0)
    .reset_index()
)

# Ensure column order
for c in ["dropoff", "recovery", "pickup"]:
    if c not in slot_bus_wide.columns:
        slot_bus_wide[c] = 0

# Plot
fig_slot_stack = px.bar(
    slot_bus_wide,
    x="slot_label",
    y=["dropoff", "recovery", "pickup"],
    barmode="stack",
    title=None
)
fig_slot_stack.update_layout(
    xaxis_title="Slot start (HH:MM)",
    yaxis_title="Number of buses",
    legend_title="Status",
    height=420,
    margin=dict(l=10, r=10, t=10, b=10)
)
fig_slot_stack.update_xaxes(tickangle=-45, nticks=18)

st.plotly_chart(fig_slot_stack, width='content')


# -----------------------
# JSON viewer (improved)
# -----------------------
st.subheader("Per-minute details")

# Build a minute key and sort options by total descending (busiest first)
df_t_view = df_t.copy()
df_t_view["time_key"] = (
    df_t_view["time_hhmm"].astype(str)
    + " (day+" + df_t_view["day_offset"].astype(str) + ")"
    + " | total=" + df_t_view["total"].astype(int).astype(str)
)

# Sort by total desc, then time_sec asc for stable ordering
df_t_view = df_t_view.sort_values(["total", "time_sec"], ascending=[False, True]).reset_index(drop=True)

sel_key = st.selectbox(
    "Pick a minute to inspect (sorted by total buses, high → low)",
    options=df_t_view["time_key"].tolist(),
    index=0
)

sel_row = df_t_view[df_t_view["time_key"] == sel_key].iloc[0]

st.write(
    f"**Selected minute:** {sel_row['time_hhmm']}  | day_offset={int(sel_row['day_offset'])} "
    f"| total={int(sel_row['total'])} "
    f"(dropoff={int(sel_row['dropoff'])}, recovery={int(sel_row['recovery'])}, pickup={int(sel_row['pickup'])})"
)

if not has_desc:
    st.info("No `description_json` column found in this CSV.")
else:
    desc_obj = parse_desc_json(sel_row["description_json"])

    # Prepare exploded DataFrames for each status
    df_drop = explode_desc(desc_obj, "dropoff")
    df_rec  = explode_desc(desc_obj, "recovery")
    df_pick = explode_desc(desc_obj, "pickup")

    # ---- NEW: total = union of all three, grouped and summed by the same fields
    base_cols = ["line_group", "block_number", "asset_class", "depot_code", "prev_route", "next_route", "n"]

    df_total = pd.concat([df_drop, df_rec, df_pick], ignore_index=True)
    if not df_total.empty:
        # ensure required columns exist
        for c in base_cols:
            if c not in df_total.columns:
                df_total[c] = None
        df_total["n"] = pd.to_numeric(df_total["n"], errors="coerce").fillna(0).astype(int)

        # group to avoid duplicates across statuses (should not happen, but safe)
        df_total = (
            df_total.groupby(
                ["line_group", "block_number", "asset_class", "depot_code", "prev_route", "next_route"],
                as_index=False,
                dropna=False
            )
            .agg(n=("n", "sum"))
        )
        df_total = df_total[base_cols]
    else:
        df_total = pd.DataFrame(columns=base_cols)

    # Asset-class filter (global for the exploded view, now includes total)
    all_assets = sorted(
        pd.concat([df_total["asset_class"], df_drop["asset_class"], df_rec["asset_class"], df_pick["asset_class"]],
                  ignore_index=True)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(all_assets) > 0:
        asset_sel = st.multiselect(
            "Filter by asset class",
            options=all_assets,
            default=all_assets
        )
    else:
        asset_sel = []

    def apply_asset_filter(dfx: pd.DataFrame) -> pd.DataFrame:
        if dfx.empty:
            return dfx
        if not asset_sel:
            return dfx
        return dfx[dfx["asset_class"].astype(str).isin(set(asset_sel))]

    df_total_f = apply_asset_filter(df_total)
    df_drop_f  = apply_asset_filter(df_drop)
    df_rec_f   = apply_asset_filter(df_rec)
    df_pick_f  = apply_asset_filter(df_pick)

    st.markdown("##### Exploded view")
    tab_total, tab_drop, tab_rec, tab_pick = st.tabs(["total", "dropoff", "recovery", "pickup"])

    with tab_total:
        if df_total_f.empty:
            st.caption("No entries for total at this minute (after filter).")
        else:
            st.dataframe(df_total_f.sort_values("n", ascending=False), width=1000, height=400)

    with tab_drop:
        if df_drop_f.empty:
            st.caption("No entries for this status at this minute (after filter).")
        else:
            st.dataframe(df_drop_f.sort_values("n", ascending=False), width=1000, height=400)

    with tab_rec:
        if df_rec_f.empty:
            st.caption("No entries for this status at this minute (after filter).")
        else:
            st.dataframe(df_rec_f.sort_values("n", ascending=False), width=1000, height=400)

    with tab_pick:
        if df_pick_f.empty:
            st.caption("No entries for this status at this minute (after filter).")
        else:
            st.dataframe(df_pick_f.sort_values("n", ascending=False), width=1000, height=400)



# -----------------------
# Raw table (still show JSON column)
# -----------------------
st.subheader("Raw minute table")
show_cols = ["time_hhmm", "day_offset", "is_next_day", "total", "dropoff", "recovery", "pickup"]
if has_desc:
    show_cols.append("description_json")

st.dataframe(
    df_t[show_cols].sort_values(["day_offset", "time_hhmm"]),
    width='content',
    height=420
)
