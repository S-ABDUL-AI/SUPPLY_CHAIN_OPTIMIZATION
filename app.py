import json
import os

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(
    page_title="Vehicle Routing — Optimization Console",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _dataframe_wide(df, **kwargs) -> None:
    """Prefer width='stretch' (Streamlit ≥1.43); fall back for older int-only width APIs."""
    kwargs.setdefault("hide_index", True)
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def _pydeck_wide(deck) -> None:
    try:
        st.pydeck_chart(deck, width="stretch")
    except TypeError:
        st.pydeck_chart(deck, use_container_width=True)


def _button_wide(button_fn, label: str, **kwargs) -> bool:
    try:
        return bool(button_fn(label, width="stretch", **kwargs))
    except TypeError:
        return bool(button_fn(label, use_container_width=True, **kwargs))


CO2_KG_PER_KM = 0.27
BUILD_VERSION = "vr-2026-04-16-2"
PALETTE = [
    [239, 68, 68],
    [34, 197, 94],
    [59, 130, 246],
    [234, 179, 8],
    [168, 85, 247],
    [20, 184, 166],
    [244, 114, 182],
]

_TRUST_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }
    .block-container { padding-top: 1rem; max-width: 100%; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; color: #253858; }
    h1 { color: #0052CC !important; font-weight: 700 !important; }
    h2, h3 { color: #253858 !important; }
    .vr-insight-box {
        border-radius: 12px;
        padding: 20px 22px;
        margin: 14px 0 20px 0;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
        border-left-width: 5px;
        border-left-style: solid;
    }
    .vr-insight-kicker { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
    .vr-insight-lead { color: #253858; font-size: 1.15rem; font-weight: 800; line-height: 1.35; margin: 10px 0 12px 0; }
    .vr-insight-body { color: #334155; font-size: 0.98rem; line-height: 1.55; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
</style>
"""
st.markdown(_TRUST_CSS, unsafe_allow_html=True)


def normalize_routes_df(raw: pd.DataFrame) -> pd.DataFrame:
    colmap = {c.lower().strip(): c for c in raw.columns}
    lat_c = next((colmap[k] for k in ("lat", "latitude") if k in colmap), None)
    lon_c = next((colmap[k] for k in ("lon", "lng", "longitude") if k in colmap), None)
    dem_c = colmap.get("demand")
    id_c = colmap.get("id")
    if not all([lat_c, lon_c, dem_c]):
        raise ValueError("CSV must include lat/latitude, lon/longitude, and demand.")
    out = pd.DataFrame(
        {
            "id": raw[id_c] if id_c else range(len(raw)),
            "lat": pd.to_numeric(raw[lat_c], errors="coerce"),
            "lon": pd.to_numeric(raw[lon_c], errors="coerce"),
            "demand": pd.to_numeric(raw[dem_c], errors="coerce").fillna(0),
        }
    )
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    out["demand"] = np.maximum(np.round(out["demand"]).astype(int), 0)
    if out.empty:
        raise ValueError("No valid coordinates after parsing.")
    out["id"] = out["id"].astype(str)
    return out


@st.cache_data(show_spinner=False)
def load_default_routes(path: str = "routes.csv") -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def euclidean_meters_matrix(locations: list[tuple[float, float]]) -> dict[int, dict[int, int]]:
    n = len(locations)
    mat: dict[int, dict[int, int]] = {}
    for i in range(n):
        mat[i] = {}
        for j in range(n):
            if i == j:
                mat[i][j] = 0
            else:
                d = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))
                mat[i][j] = int(d * 111_000)
    return mat


def build_naive_manual_routes(
    locations: list[tuple[float, float]],
    demands: list[int],
    depot_idx: int,
    num_vehicles: int,
    vehicle_capacity: int,
    distance_matrix: dict[int, dict[int, int]],
) -> dict:
    customer_nodes = [i for i in range(len(locations)) if i != depot_idx]
    groups = [customer_nodes[i::num_vehicles] for i in range(num_vehicles)]
    route_paths: list[dict] = []
    total_distance = 0
    used = 0
    for v, nodes in enumerate(groups, start=1):
        if not nodes:
            continue
        used += 1
        route = [depot_idx] + nodes + [depot_idx]
        load = sum(demands[n] for n in nodes)
        distance_m = 0
        for a, b in zip(route[:-1], route[1:]):
            distance_m += distance_matrix[a][b]
        total_distance += distance_m
        route_paths.append(
            {
                "vehicle_id": v,
                "nodes": route,
                "route": [(locations[n][0], locations[n][1]) for n in route],
                "load": load,
                "distance_m": distance_m,
                "capacity_ratio": (load / max(vehicle_capacity, 1)),
            }
        )
    if used == 0:
        avg_util = 0.0
    else:
        avg_util = float(np.mean([min(1.0, r["capacity_ratio"]) for r in route_paths])) * 100.0
    return {
        "routes": route_paths,
        "total_distance_m": total_distance,
        "vehicles_used": used,
        "avg_util_pct": avg_util,
    }


def solve_cvrp(
    locations: list[tuple[float, float]],
    demands: list[int],
    depot_idx: int,
    num_vehicles: int,
    vehicle_capacity: int,
    distance_matrix: dict[int, dict[int, int]],
    solver_time: int,
) -> dict | None:
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_cb = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,
        [vehicle_capacity] * num_vehicles,
        True,
        "Capacity",
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = int(solver_time)

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None

    route_paths = []
    total_distance = 0
    used = 0
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        node_path = [manager.IndexToNode(index)]
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_load += demands[node]
            prev = index
            index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(index)
            route_distance += routing.GetArcCostForVehicle(prev, index, vehicle_id)
            node_path.append(next_node)
        unique_customers = [n for n in node_path if n != depot_idx]
        if unique_customers:
            used += 1
        route_paths.append(
            {
                "vehicle_id": vehicle_id + 1,
                "nodes": node_path,
                "route": [(locations[n][0], locations[n][1]) for n in node_path],
                "load": route_load,
                "distance_m": route_distance,
                "capacity_ratio": (route_load / max(vehicle_capacity, 1)),
            }
        )
        total_distance += route_distance
    avg_util_pct = float(np.mean([min(1.0, r["capacity_ratio"]) for r in route_paths])) * 100.0 if route_paths else 0.0
    return {
        "routes": route_paths,
        "total_distance_m": total_distance,
        "vehicles_used": used,
        "avg_util_pct": avg_util_pct,
    }


def build_route_map(
    route_paths: list[dict],
    all_locations: list[tuple[float, float]],
    title: str,
) -> pdk.Deck:
    path_rows = []
    stop_rows = []
    seq_rows = []
    for i, r in enumerate(route_paths):
        color = PALETTE[i % len(PALETTE)]
        path_rows.append(
            {
                "vehicle_id": r["vehicle_id"],
                "path": [[lon, lat] for lat, lon in r["route"]],
                "color": color,
                "distance_km": r["distance_m"] / 1000.0,
            }
        )
        for seq, n in enumerate(r["nodes"]):
            lat, lon = all_locations[n]
            stop_rows.append(
                {
                    "vehicle_id": r["vehicle_id"],
                    "lat": lat,
                    "lon": lon,
                    "seq": seq,
                    "radius": 80 if seq == 0 else 45,
                    "color": color,
                }
            )
            if seq > 0:
                seq_rows.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "label": f"{seq}",
                        "color": color,
                    }
                )

    layers = [
        pdk.Layer(
            "PathLayer",
            data=path_rows,
            get_path="path",
            get_color="color",
            width_scale=8,
            width_min_pixels=4,
            pickable=True,
            auto_highlight=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=stop_rows,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=4,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=seq_rows,
            get_position="[lon, lat]",
            get_text="label",
            get_color="[255,255,255]",
            get_size=13,
            get_alignment_baseline="'center'",
            pickable=False,
        ),
    ]
    mid_lat = float(np.mean([lat for lat, _ in all_locations]))
    mid_lon = float(np.mean([lon for _, lon in all_locations]))
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=11, pitch=35),
        tooltip={"text": f"{title}\nVehicle {{vehicle_id}}\nSeq {{seq}}"},
        # Carto GL style works without a Mapbox token (Mapbox URLs often fail on Streamlit Cloud).
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )


def build_export_df(route_paths: list[dict], all_locations: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for r in route_paths:
        for seq, n in enumerate(r["nodes"]):
            lat, lon = all_locations[n]
            rows.append(
                {
                    "vehicle_id": r["vehicle_id"],
                    "seq": seq,
                    "node": n,
                    "lat": lat,
                    "lon": lon,
                }
            )
    return pd.DataFrame(rows)


st.title("Vehicle Routing — Optimization Console")
st.caption(
    "Minimize route distance under capacity constraints using OR-Tools (CVRP). "
    "This interface is structured for scenario planning and executive readouts."
)
st.caption(f"Build: `{BUILD_VERSION}`")

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader(
    "Upload stops CSV (`id`, `lat`/`lon`, `demand`)",
    type=["csv"],
)
if uploaded is not None:
    try:
        data = normalize_routes_df(pd.read_csv(uploaded))
        st.sidebar.success("Custom route file loaded.")
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    raw = load_default_routes()
    if raw is None:
        st.error("Default routes.csv not found in app directory.")
        st.stop()
    data = normalize_routes_df(raw)
    st.sidebar.info("Using bundled routes.csv.")

st.sidebar.header("What-if (stress test)")
demand_multiplier = st.sidebar.slider(
    "Demand multiplier ×",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.05,
    help="Scales all customer demand before baseline + optimization.",
)

st.sidebar.header("Solver")
num_vehicles = st.sidebar.number_input("Vehicles", min_value=1, max_value=20, value=3)
vehicle_capacity = st.sidebar.number_input("Capacity per vehicle (units)", min_value=1, max_value=10_000, value=30)
solver_time = st.sidebar.slider("Time limit (seconds)", 5, 120, 15)

depot_options = [f"{idx}: {row['id']}" for idx, row in data.iterrows()]
depot_pick = st.sidebar.selectbox("Depot node", options=depot_options, index=0)
depot_idx = int(depot_pick.split(":")[0])

compare_mode = st.sidebar.toggle("Compare baseline vs optimized map", value=False)
run_sidebar = _button_wide(st.sidebar.button, "Run optimization", type="primary")

st.sidebar.divider()
st.sidebar.subheader("About")
st.sidebar.markdown(
    "**Sherriff Abdul-Hamid**  \n"
    "[GitHub](https://github.com/S-ABDUL-AI) · "
    "[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)"
)

working = data.copy()
working["demand"] = np.maximum(np.round(working["demand"] * demand_multiplier).astype(int), 0)
locations = list(zip(working["lat"], working["lon"]))
demands = working["demand"].tolist()
distance_matrix = euclidean_meters_matrix(locations)

manual = build_naive_manual_routes(
    locations=locations,
    demands=demands,
    depot_idx=depot_idx,
    num_vehicles=int(num_vehicles),
    vehicle_capacity=int(vehicle_capacity),
    distance_matrix=distance_matrix,
)

total_stops = max(0, len(working) - 1)
total_demand = int(np.sum(demands) - demands[depot_idx])
fleet_capacity = int(num_vehicles * vehicle_capacity)
baseline_km = manual["total_distance_m"] / 1000.0

s1, s2, s3, s4 = st.columns(4)
s1.metric("Total stops", f"{total_stops:,}", delta=f"Depot node {depot_idx}", delta_color="off")
s2.metric("Total demand (units)", f"{total_demand:,}", delta=f"×{demand_multiplier:.2f} demand", delta_color="off")
s3.metric("Fleet capacity (units)", f"{fleet_capacity:,}", delta=f"{num_vehicles} vehicles", delta_color="off")
s4.metric("Baseline distance (km)", f"{baseline_km:,.2f}", delta="Manual split policy", delta_color="off")

st.divider()
st.subheader("Input parameters")
st.caption("Data and assumptions feeding the baseline + optimized routing run.")
c1, c2 = st.columns([2, 1])
with c1:
    _dataframe_wide(working.head(30))
with c2:
    st.markdown("**Scenario settings**")
    st.write(f"- Vehicles: **{num_vehicles}**")
    st.write(f"- Capacity/vehicle: **{vehicle_capacity}**")
    st.write(f"- Depot node: **{depot_idx}**")
    st.write(f"- Solver time limit: **{solver_time}s**")

with st.expander("Objective & constraints (algorithm audit)", expanded=False):
    st.markdown(
        """
**Decision variables**  
- Route arcs and customer assignments across vehicles.

**Objective (minimize)**  
- Total route distance across all vehicle tours.

**Constraints**  
- Every customer is visited exactly once.  
- Each route starts/ends at the selected depot.  
- Sum of assigned demand on a vehicle route does not exceed vehicle capacity.

**Baseline policy (for deltas)**  
- Customers are split across vehicles in round-robin order, then each vehicle returns to depot.
        """
    )

with st.expander("Technical methodology & assumptions"):
    st.markdown(
        """
- **Model:** Capacitated Vehicle Routing Problem (CVRP).  
- **Solver:** Google OR-Tools routing with cheapest-arc initialization + guided local search refinement.  
- **Distance model:** Euclidean approximation in meters (`~111,000 m/degree`) for demonstration.  
- **CO2 proxy:** fixed factor of `0.27 kg/km` on distance reductions vs baseline.
        """
    )

st.divider()
run_main = _button_wide(st.button, "Run optimization", type="primary", key="run_main_vr")
run_opt = run_sidebar or run_main

st.subheader("Visual analysis")
if not run_opt:
    st.info("Configure assumptions in the sidebar, then click **Run optimization** (sidebar or the primary button above).")
    st.stop()

optimized = solve_cvrp(
    locations=locations,
    demands=demands,
    depot_idx=depot_idx,
    num_vehicles=int(num_vehicles),
    vehicle_capacity=int(vehicle_capacity),
    distance_matrix=distance_matrix,
    solver_time=int(solver_time),
)
if optimized is None:
    st.warning("No feasible solution — raise capacity, add vehicles, or reduce demand multiplier.")
    st.stop()

opt_km = optimized["total_distance_m"] / 1000.0
delta_km = baseline_km - opt_km
co2_saved_kg = max(0.0, delta_km) * CO2_KG_PER_KM
util_pct = optimized["avg_util_pct"]
max_load = int(max((r["load"] for r in optimized["routes"]), default=0))

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Optimized distance (km)",
    f"{opt_km:,.2f}",
    delta=f"{delta_km:+.2f} km vs baseline",
    delta_color="normal" if delta_km >= 0 else "inverse",
)
k2.metric("Vehicles used", optimized["vehicles_used"], delta=f"Configured {num_vehicles}", delta_color="off")
k3.metric("Avg utilization", f"{util_pct:.1f}%", delta=f"Peak load {max_load}", delta_color="off")
k4.metric("CO2 saved", f"{co2_saved_kg:,.2f} kg", delta=f"{max(0.0, delta_km):.2f} km avoided", delta_color="off")

st.subheader("Scenario comparison — distance")
d1, d2 = st.columns(2)
with d1:
    st.metric("Baseline route distance", f"{baseline_km:,.2f} km")
with d2:
    st.metric(
        "Optimized route distance",
        f"{opt_km:,.2f} km",
        delta=f"{delta_km:+.2f} km improvement",
        delta_color="normal" if delta_km >= 0 else "inverse",
    )

if delta_km > 0:
    accent = "#0d9488"
    lead = (
        f"<strong>Approve optimized dispatch</strong> — routing reduces distance by "
        f"<strong>{delta_km:,.2f} km</strong> (~<strong>{co2_saved_kg:,.2f} kg CO2</strong> avoided)."
    )
    body = (
        f"Average vehicle utilization is <strong>{util_pct:.1f}%</strong>. "
        "Export route artifacts for dispatch execution and rerun with alternate demand multipliers for stress testing."
    )
else:
    accent = "#64748b"
    lead = "<strong>Neutral scenario</strong> — optimized and baseline distances are nearly identical at current assumptions."
    body = (
        "Try increasing solver time, changing the depot node, or altering fleet capacity to expose more separable routing gains."
    )

st.markdown(
    f"""
<div class="vr-insight-box" style="border-left-color:{accent};">
  <div class="vr-insight-kicker" style="color:{accent};">Executive insight</div>
  <div class="vr-insight-lead">{lead}</div>
  <div class="vr-insight-body">{body}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("Map")
if compare_mode:
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**Baseline (manual split)**")
        _pydeck_wide(build_route_map(manual["routes"], locations, "Baseline route"))
    with mc2:
        st.markdown("**Optimized (OR-Tools)**")
        _pydeck_wide(build_route_map(optimized["routes"], locations, "Optimized route"))
else:
    _pydeck_wide(build_route_map(optimized["routes"], locations, "Optimized route"))

rows = []
for r in optimized["routes"]:
    customer_count = len([n for n in r["nodes"] if n != depot_idx])
    rows.append(
        {
            "vehicle": r["vehicle_id"],
            "stops": customer_count,
            "load": int(r["load"]),
            "distance_km": round(r["distance_m"] / 1000.0, 3),
            "capacity_util_pct": round(min(1.0, r["capacity_ratio"]) * 100.0, 1),
        }
    )
details_df = pd.DataFrame(rows).sort_values("vehicle")
st.markdown("#### Route-level output")
_dataframe_wide(details_df)

export_df = build_export_df(optimized["routes"], locations)
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download routes CSV",
        export_df.to_csv(index=False),
        "optimized_routes.csv",
        "text/csv",
    )
with c2:
    st.download_button(
        "Download routes JSON",
        json.dumps(optimized["routes"], indent=2),
        "optimized_routes.json",
        "application/json",
    )
