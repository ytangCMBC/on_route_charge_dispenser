
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from heapq import heappush, heappop

# =========================
# Config
# =========================
EVENTS_PATH = "output_term/charging_event_record_test.csv"
DISPENSERS_PATH = "output_term/dispensers_needed_by_candidate.csv"

MIN_SESSION_MINUTES = 0 
BASE_DAY = pd.Timestamp("2025-09-15") 


# =========================
# Helpers
# =========================
def _require_cols(df: pd.DataFrame, cols: set, name: str):
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _format_hour_window(h: int) -> str:
    return f"{h}:00:00 - {h+1}:00:00"


def _format_minute_label(t0_sec: int, minute_idx: int) -> str:
    """Label minutes as H:MM where H can exceed 24."""
    t = t0_sec + minute_idx * 60
    h = t // 3600
    m = (t % 3600) // 60
    return f"{int(h)}:{int(m):02d}"


def _build_constant_segments(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build constant-concurrency segments via sweep line.
    Half-open intervals: [start, end)
    Tie-break: END before START at the same timestamp.
    Returns segments with columns: seg_start, seg_end, conc
    """
    if events.empty:
        return pd.DataFrame(columns=["seg_start", "seg_end", "conc"])

    starts = events["start_sec"].to_numpy(dtype=float)
    ends = events["end_sec"].to_numpy(dtype=float)

    marks = []
    for s, e in zip(starts, ends):
        marks.append((float(s), +1, 1))  # start
        marks.append((float(e), -1, 0))  # end (processed first at same time)

    marks.sort(key=lambda x: (x[0], x[2]))

    cur = 0
    prev_t = None
    segs = []

    for t, delta, _ in marks:
        if prev_t is not None and t > prev_t:
            if cur > 0:
                segs.append((prev_t, t, cur))
        cur += delta
        prev_t = t

    return pd.DataFrame(segs, columns=["seg_start", "seg_end", "conc"])


def _minute_peak_busyness_any_window(events: pd.DataFrame, window_start_sec: int, window_end_sec: int) -> pd.DataFrame:
    """
    Per-minute PEAK concurrent sessions for a window [window_start, window_end),
    using constant segments (fast, matches Gantt logic).
    """
    if events.empty or window_end_sec <= window_start_sec:
        return pd.DataFrame({"minute_idx": np.array([], dtype=int), "active_sessions": np.array([], dtype=int)})

    ev = events[(events["start_sec"] < window_end_sec) & (events["end_sec"] > window_start_sec)].copy()
    n_minutes = int(np.ceil((window_end_sec - window_start_sec) / 60.0))

    if ev.empty:
        return pd.DataFrame({"minute_idx": np.arange(n_minutes, dtype=int), "active_sessions": np.zeros(n_minutes, dtype=int)})

    ev["start_sec"] = ev["start_sec"].clip(lower=window_start_sec, upper=window_end_sec)
    ev["end_sec"] = ev["end_sec"].clip(lower=window_start_sec, upper=window_end_sec)

    segs = _build_constant_segments(ev)
    peaks = np.zeros(n_minutes, dtype=int)

    if segs.empty:
        return pd.DataFrame({"minute_idx": np.arange(n_minutes, dtype=int), "active_sessions": peaks})

    t0 = float(window_start_sec)

    for a, b, c in segs.itertuples(index=False):
        i0 = int(np.floor((a - t0) / 60.0))
        i1 = int(np.ceil((b - t0) / 60.0)) - 1
        if i1 < 0 or i0 >= n_minutes:
            continue
        i0 = max(i0, 0)
        i1 = min(i1, n_minutes - 1)
        if i0 <= i1:
            peaks[i0:i1 + 1] = np.maximum(peaks[i0:i1 + 1], int(c))

    return pd.DataFrame({"minute_idx": np.arange(n_minutes, dtype=int), "active_sessions": peaks})


def _hour_stats(events: pd.DataFrame) -> pd.DataFrame:
    """
    Hourly windows ranked by:
      1) peak concurrent sessions (within-minute peak)
      2) total of per-minute peaks (proxy of intensity)
      3) number of sessions that overlap the hour
    """
    if events.empty:
        return pd.DataFrame(columns=["hour", "label", "peak_active", "total_active_sessions_minutes", "sessions_overlap"])

    h_min = int(np.floor(events["start_sec"].min() / 3600.0))
    h_max = int(np.ceil(events["end_sec"].max() / 3600.0))  # exclusive upper
    hours = list(range(h_min, h_max))

    rows = []
    for h in hours:
        hs = h * 3600
        he = hs + 3600
        sub = events[(events["start_sec"] < he) & (events["end_sec"] > hs)]
        minute_df = _minute_peak_busyness_any_window(sub, hs, he)
        peak_active = int(minute_df["active_sessions"].max()) if not minute_df.empty else 0
        total_active = int(minute_df["active_sessions"].sum()) if not minute_df.empty else 0

        rows.append(
            {
                "hour": h,
                "label": _format_hour_window(h),
                "peak_active": peak_active,
                "total_active_sessions_minutes": total_active,
                "sessions_overlap": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _build_block_label(df: pd.DataFrame) -> pd.Series:
    if "line_group" in df.columns and "block_number" in df.columns:
        return "LG " + df["line_group"].astype(str) + "-BLK " + df["block_number"].astype(str)
    return df["block_id"].astype(str)


def _coverage_curve_max_sessions(events: pd.DataFrame, max_k: int) -> pd.DataFrame:
    """
    For k=1..max_k, compute the maximum % of sessions that can be served
    without overlap conflicts on k dispensers (k-parallel scheduling).

    Half-open intervals: [start, end) so end at t does not block start at t.
    """
    if events.empty or max_k <= 0:
        return pd.DataFrame({"dispensers": [], "covered_sessions": [], "coverage_pct": []})

    df = events.loc[:, ["start_sec", "end_sec"]].copy()
    df["start_sec"] = df["start_sec"].astype(float)
    df["end_sec"] = df["end_sec"].astype(float)
    df = df[df["end_sec"] > df["start_sec"]].sort_values(["start_sec", "end_sec"]).reset_index(drop=True)

    total = int(len(df))
    if total == 0:
        return pd.DataFrame({"dispensers": list(range(1, max_k + 1)),
                             "covered_sessions": [0]*max_k,
                             "coverage_pct": [0.0]*max_k})

    starts = df["start_sec"].to_numpy()
    ends = df["end_sec"].to_numpy()

    rows = []
    for k in range(1, max_k + 1):
        active_ends = []  # sorted active end times
        kept = 0

        for s, e in zip(starts, ends):
            while active_ends and active_ends[0] <= s:
                active_ends.pop(0)

            # insert end into sorted list
            lo, hi = 0, len(active_ends)
            while lo < hi:
                mid = (lo + hi) // 2
                if active_ends[mid] < e:
                    lo = mid + 1
                else:
                    hi = mid
            active_ends.insert(lo, e)

            if len(active_ends) > k:
                active_ends.pop()  # drop latest end
            else:
                kept += 1

        rows.append({"dispensers": k, "covered_sessions": kept, "coverage_pct": 100.0 * kept / total})

    return pd.DataFrame(rows)



def _peak_concurrency(events: pd.DataFrame) -> int:
    """Peak concurrent sessions using half-open intervals [start, end)."""
    if events.empty:
        return 0
    marks = []
    for s, e in zip(events["start_sec"].to_numpy(dtype=float), events["end_sec"].to_numpy(dtype=float)):
        marks.append((float(s), +1, 1))  # start
        marks.append((float(e), -1, 0))  # end first at same timestamp
    marks.sort(key=lambda x: (x[0], x[2]))
    cur = 0
    mx = 0
    for _, d, _ in marks:
        cur += d
        mx = max(mx, cur)
    return int(mx)


def _fcfs_blocks_supported(events: pd.DataFrame, k: int) -> tuple[int, int]:
    """
    FCFS scheduling on k dispensers.
    If any session of a block cannot be scheduled, that block is NOT fully supported.
    Returns: (supported_blocks, total_blocks)
    """
    if events.empty or k <= 0:
        return 0, int(events["block_id"].nunique()) if "block_id" in events.columns else 0

    df = events.loc[:, ["block_id", "start_sec", "end_sec"]].copy()
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
    df = df.dropna(subset=["block_id", "start_sec", "end_sec"])
    df = df[df["end_sec"] > df["start_sec"]].copy()

    all_blocks = set(df["block_id"].astype(str).unique())
    total_blocks = len(all_blocks)
    if total_blocks == 0:
        return 0, 0

    # stable sort so ties are deterministic
    df["block_id"] = df["block_id"].astype(str)
    df = df.sort_values(["start_sec", "end_sec", "block_id"], kind="mergesort")

    # min-heap of dispenser end times
    heap = []
    dropped_blocks = set()

    for r in df.itertuples(index=False):
        s = float(r.start_sec)
        e = float(r.end_sec)
        b = r.block_id

        # free dispensers that finished by s
        while heap and heap[0] <= s:
            heappop(heap)

        if len(heap) < k:
            heappush(heap, e)
        else:
            dropped_blocks.add(b)

    supported_blocks = len(all_blocks - dropped_blocks)
    return supported_blocks, total_blocks


def _coverage_curve_blocks_fcfs(events: pd.DataFrame, max_k: int) -> pd.DataFrame:
    """k=1..max_k => % blocks fully supported under FCFS."""
    rows = []
    for k in range(1, max_k + 1):
        sup, tot = _fcfs_blocks_supported(events, k)
        rows.append(
            {
                "dispensers": k,
                "blocks_supported": sup,
                "blocks_total": tot,
                "coverage_pct_blocks": (100.0 * sup / tot) if tot else np.nan,
            }
        )
    return pd.DataFrame(rows)

# =========================
# App
# =========================
st.set_page_config(page_title="On-route charging — Dispenser Explorer", layout="wide")
st.title("On-route charging — Dispenser Explorer")

events_df = pd.read_csv(EVENTS_PATH)
disp_df = pd.read_csv(DISPENSERS_PATH)

_require_cols(events_df, {"candidate_name", "start_sec", "end_sec", "block_id"}, "charging_event_record")

events_df = events_df.copy()
events_df["candidate_name"] = events_df["candidate_name"].astype(str)
events_df["start_sec"] = pd.to_numeric(events_df["start_sec"], errors="coerce")
events_df["end_sec"] = pd.to_numeric(events_df["end_sec"], errors="coerce")
events_df["block_id"] = events_df["block_id"].astype(str)

events_df = events_df.dropna(subset=["candidate_name", "start_sec", "end_sec"])
events_df = events_df[events_df["end_sec"] > events_df["start_sec"]].copy()

if MIN_SESSION_MINUTES and MIN_SESSION_MINUTES > 0:
    min_sec = MIN_SESSION_MINUTES * 60.0
    events_df = events_df[(events_df["end_sec"] - events_df["start_sec"]) >= min_sec].copy()

# Dispensers file columns
disp_cols = set(disp_df.columns)
cand_col = "candidate_name" if "candidate_name" in disp_cols else None
need_col = None
for c in ["dispensers_needed", "dispenser_needed", "num_dispensers", "dispensers", "needed"]:
    if c in disp_cols:
        need_col = c
        break

if cand_col is None or need_col is None:
    st.error(
        "dispensers_needed_by_candidate.csv must contain columns like "
        "`candidate_name` and one of: dispensers_needed / num_dispensers / dispensers / needed."
    )
    st.stop()

disp_df = disp_df.copy()
disp_df[cand_col] = disp_df[cand_col].astype(str)
disp_df[need_col] = pd.to_numeric(disp_df[need_col], errors="coerce").fillna(0).astype(int)
need_map = dict(zip(disp_df[cand_col], disp_df[need_col]))

# Candidate selector
candidates = sorted(events_df["candidate_name"].unique().tolist())
if not candidates:
    st.warning("No charging sessions found in the events table.")
    st.stop()

sel_candidate = st.selectbox("Select a candidate on-route charging location", candidates, index=0)
cand_events = events_df[events_df["candidate_name"] == str(sel_candidate)].copy()

disp_needed = int(need_map.get(str(sel_candidate), 0))
total_sessions = int(len(cand_events))
unique_blocks = int(cand_events["block_id"].nunique())

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Dispensers needed (plan)", disp_needed)
kpi2.metric("Charging sessions (all day)", total_sessions)
kpi3.metric("Unique blocks (all day)", unique_blocks)


st.markdown("### Block Coverage vs. number of dispensers (block fully supported)")

if cand_events.empty:
    st.info("No sessions for this candidate.")
else:
    # choose max_k: up to plan
    max_k = max(1, disp_needed)

    max_k = min(int(max_k), 40)  # guardrail (adjust as you like)

    curve_blocks = _coverage_curve_blocks_fcfs(cand_events, max_k=max_k)

    fig_cov_blocks = px.line(
        curve_blocks,
        x="dispensers",
        y="coverage_pct_blocks",
        markers=True,
        title=None,
        hover_data={"blocks_supported": True, "blocks_total": True, "coverage_pct_blocks": ":.2f"},
    )
    fig_cov_blocks.update_layout(
        xaxis_title="Number of dispensers at this location",
        yaxis_title="% of blocks fully supported (all sessions served)",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 102]),
    )
    fig_cov_blocks.update_xaxes(dtick=1)

    if disp_needed > 0:
        fig_cov_blocks.add_vline(
            x=disp_needed,
            line_dash="dash",
            annotation_text="Plan",
            annotation_position="top left",
        )

    st.plotly_chart(fig_cov_blocks, width="stretch")

# =========================
# Coverage vs. dispensers curve (all-day, by definition)
# =========================
st.markdown("### Charging Session Coverage vs. number of dispensers")
if cand_events.empty:
    st.info("No sessions for this candidate.")
else:
    t0_all = int(np.floor(cand_events["start_sec"].min() / 60.0) * 60)
    t1_all = int(np.ceil(cand_events["end_sec"].max() / 60.0) * 60)
    peak_conc_all = int(_minute_peak_busyness_any_window(cand_events, t0_all, t1_all)["active_sessions"].max())
    max_k = max(peak_conc_all, 1)
    max_k = min(int(max_k), 30)  # guardrail; raise if needed

    curve_df = _coverage_curve_max_sessions(cand_events, max_k=max_k)

    fig_cov = px.line(curve_df, x="dispensers", y="coverage_pct", markers=True, title=None)
    fig_cov.update_layout(
        xaxis_title="Number of dispensers at this location",
        yaxis_title="% of charging sessions that can be covered",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 102]),
    )
    fig_cov.update_xaxes(dtick=1)
    if disp_needed > 0:
        fig_cov.add_vline(x=disp_needed, line_dash="dash", annotation_text="Plan", annotation_position="top left")
    st.plotly_chart(fig_cov, width="stretch")

st.divider()

# =========================
# Hour selection (NO all-day view for gantt/busyness)
# =========================
hour_tbl = _hour_stats(cand_events)
if hour_tbl.empty:
    st.warning("No valid hourly windows for this candidate.")
    st.stop()

hour_tbl = hour_tbl.sort_values(
    ["peak_active", "total_active_sessions_minutes", "sessions_overlap", "hour"],
    ascending=[False, False, False, True],
).reset_index(drop=True)

sel_label = st.selectbox(
    "Select an hourly time window (sorted by busiest → emptiest)",
    hour_tbl["label"].tolist(),
    index=0,  # default: busiest hour
)

sel_hour = int(hour_tbl.loc[hour_tbl["label"] == sel_label, "hour"].iloc[0])
window_start = sel_hour * 3600
window_end = window_start + 3600
window_title = sel_label

win_events = cand_events[(cand_events["start_sec"] < window_end) & (cand_events["end_sec"] > window_start)].copy()


minute_df = _minute_peak_busyness_any_window(win_events, window_start, window_end)
peak_active = int(minute_df["active_sessions"].max()) if not minute_df.empty else 0


# =========================
# Gantt (hour only)
# =========================
st.markdown(f"### Gantt — {window_title}")

if win_events.empty:
    st.info("No sessions overlap the selected hour.")
else:
    plot_df = win_events.copy()
    plot_df["clip_start"] = plot_df["start_sec"].clip(lower=window_start, upper=window_end)
    plot_df["clip_end"] = plot_df["end_sec"].clip(lower=window_start, upper=window_end)
    plot_df["block_label"] = _build_block_label(plot_df)

    plot_df["start_ts"] = BASE_DAY + pd.to_timedelta(plot_df["clip_start"].astype(float), unit="s")
    plot_df["end_ts"] = BASE_DAY + pd.to_timedelta(plot_df["clip_end"].astype(float), unit="s")

    order = (
        plot_df.groupby("block_label")["start_ts"]
        .min()
        .sort_values()
        .index
        .tolist()
    )
    plot_df["block_label"] = pd.Categorical(plot_df["block_label"], categories=order, ordered=True)

    hover_cols = [c for c in [
        "block_id", "line_group", "block_number",
        "depot_code", "event_type",
        "soc_start_pct", "soc_end_pct", "charged_kwh",
        "prev_route_short_name", "next_route_short_name",
        "prev_trip_end_stop_name", "next_trip_start_stop_name",
        "start_sec", "end_sec",
    ] if c in plot_df.columns]

    color_col = "depot_code" if "depot_code" in plot_df.columns else None

    fig_gantt = px.timeline(
        plot_df,
        x_start="start_ts",
        x_end="end_ts",
        y="block_label",
        color=color_col,
        hover_data=hover_cols,
        title=None,
    )

    x0 = BASE_DAY + pd.to_timedelta(window_start, unit="s")
    x1 = BASE_DAY + pd.to_timedelta(window_end, unit="s")

    fig_gantt.update_layout(
        xaxis_title="Time",
        yaxis_title="Block",
        height=min(950, 140 + 22 * plot_df["block_label"].nunique()),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_gantt.update_xaxes(
        tickformat="%H:%M",
        range=[x0, x1],
        rangeslider_visible=False,
    )
    fig_gantt.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_gantt, width="stretch")

# =========================
# Busyness (hour only)
# =========================
st.markdown(f"### Busyness (per-minute peak concurrency) — {window_title}")

if minute_df.empty:
    st.info("No minutes to display in this hour.")
else:
    minute_plot = minute_df.copy()
    minute_plot["time_hhmm"] = minute_plot["minute_idx"].map(lambda i: _format_minute_label(window_start, int(i)))
    minute_plot["over_capacity"] = (minute_plot["active_sessions"] > disp_needed) if disp_needed > 0 else False

    if disp_needed > 0:
        over_minutes = int(minute_plot["over_capacity"].sum())
        st.caption(
            f"Peak concurrent sessions: **{int(minute_plot['active_sessions'].max())}** | "
            f"Minutes over capacity ({disp_needed} dispensers): **{over_minutes}/60**"
        )

    fig_busy = px.bar(
        minute_plot,
        x="minute_idx",
        y="active_sessions",
        hover_data={"time_hhmm": True, "active_sessions": True, "over_capacity": True},
        title=None,
    )
    fig_busy.update_traces(
        hovertemplate="Time: %{customdata[0]}<br>Peak concurrent sessions: %{y}<br>Over capacity: %{customdata[2]}<extra></extra>"
    )
    fig_busy.update_layout(
        xaxis_title="Minute within hour (0–59)",
        yaxis_title="Peak concurrent sessions in minute",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_busy.update_xaxes(dtick=5)

    if disp_needed > 0:
        fig_busy.add_hline(
            y=disp_needed,
            line_dash="dash",
            annotation_text=f"Capacity ({disp_needed})",
            annotation_position="top left",
        )
        ymax = max(int(minute_plot["active_sessions"].max()), disp_needed)
        fig_busy.add_hrect(
            y0=disp_needed,
            y1=ymax,
            opacity=0.08,
            line_width=0,
            annotation_text="Over capacity",
            annotation_position="top right",
        )

    st.plotly_chart(fig_busy, width="stretch")

# =========================
# Raw records (selected hour)
# =========================
st.markdown("### Raw charging-session records (selected hour)")
if win_events.empty:
    st.info("No records in the selected hour.")
else:
    raw_cols = [
        "candidate_name", "block_id", "line_group", "block_number", "depot_code",
        "start_dt", "end_dt", "event_type", "soc_start_pct", "soc_end_pct",
        "charged_kwh", "prev_route_short_name", "next_route_short_name",
        "prev_trip_end_stop_name", "next_trip_start_stop_name", "start_sec", "end_sec",
    ]
    raw_cols = [c for c in raw_cols if c in win_events.columns]
    raw_df = win_events.loc[:, raw_cols].copy()
    raw_df = raw_df.sort_values(["start_sec", "end_sec"], ascending=[True, True]).reset_index(drop=True)
    st.dataframe(raw_df, width="stretch", height=320)

with st.expander("Show hour ranking table"):
    st.dataframe(hour_tbl, width="stretch")
