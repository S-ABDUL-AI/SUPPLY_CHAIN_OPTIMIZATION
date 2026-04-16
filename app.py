import json
import os

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(
    page_title="Logistics Command Center",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


CO2_KG_PER_KM = 0.27
PALETTE = [
    [239, 68, 68],
    [34, 197, 94],
    [59, 130, 246],
    [234, 179, 8],
    [168, 85, 247],
    [20, 184, 166],
    [244, 114, 182],
]


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
        map_style="mapbox://styles/mapbox/dark-v10",
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


st.title("Logistics Command Center: CVRP Fleet Intelligence")
st.caption(
    "Solve the Capacitated Vehicle Routing Problem (CVRP) with OR-Tools and compare a manual dispatch policy "
    "against optimized routes in one operational screen."
)

uploaded = st.file_uploader(
    "Upload customer stops CSV (`id`, `lat`/`latitude`, `lon`/`longitude`, `demand`) or use bundled routes.csv",
    type=["csv"],
)
if uploaded is not None:
    try:
        data = normalize_routes_df(pd.read_csv(uploaded))
        st.success("Custom route file loaded.")
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    raw = load_default_routes()
    if raw is None:
        st.error("Default `routes.csv` is missing.")
        st.stop()
    data = normalize_routes_df(raw)
    st.info("Using bundled routes.csv.")

st.sidebar.header("Logistics Scenario")
num_vehicles = st.sidebar.number_input("Fleet size (vehicles)", min_value=1, max_value=50, value=3)
vehicle_capacity = st.sidebar.number_input("Vehicle capacity (units)", min_value=1, max_value=100_000, value=30)
solver_time = st.sidebar.slider("Optimization time limit (sec)", 3, 120, 20)
demand_multiplier = st.sidebar.slider("Demand multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.05)

depot_options = [f"{idx}: {row['id']}" for idx, row in data.iterrows()]
depot_pick = st.sidebar.selectbox("Depot location (node)", options=depot_options, index=0)
depot_idx = int(depot_pick.split(":")[0])

compare_mode = st.sidebar.toggle("Scenario comparison: Manual vs Optimized", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Directional markers: route labels show visit sequence.")
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

solve_btn = st.button("Run route optimization", type="primary", use_container_width=True)

if solve_btn:
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
        st.warning("No feasible solution found. Increase fleet size, capacity, or reduce demand multiplier.")
        st.stop()

    opt_km = optimized["total_distance_m"] / 1000.0
    man_km = manual["total_distance_m"] / 1000.0
    delta_km = man_km - opt_km

    opt_util = optimized["avg_util_pct"]
    man_util = manual["avg_util_pct"]
    util_delta = opt_util - man_util

    co2_saved_kg = max(delta_km, 0.0) * CO2_KG_PER_KM

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(
            "Total Fleet Distance",
            f"{opt_km:,.2f} km",
            delta=f"{delta_km:+.2f} km vs manual (Optimized)",
            delta_color="normal" if delta_km >= 0 else "inverse",
        )
    with k2:
        st.metric(
            "Vehicle Utilization %",
            f"{opt_util:.1f}%",
            delta=f"{util_delta:+.1f} pts vs manual",
            delta_color="normal" if util_delta >= 0 else "inverse",
        )
    with k3:
        st.metric(
            "CO2 Emissions Saved",
            f"{co2_saved_kg:,.2f} kg",
            delta=f"{max(delta_km, 0.0):.2f} km avoided",
            delta_color="normal" if co2_saved_kg > 0 else "off",
        )

    st.subheader("Route map")
    if compare_mode:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Manual Route (baseline)**")
            st.pydeck_chart(build_route_map(manual["routes"], locations, "Manual Route"), use_container_width=True)
        with c_right:
            st.markdown("**Optimized Route (OR-Tools)**")
            st.pydeck_chart(build_route_map(optimized["routes"], locations, "Optimized Route"), use_container_width=True)
    else:
        st.pydeck_chart(build_route_map(optimized["routes"], locations, "Optimized Route"), use_container_width=True)

    details = []
    for r in optimized["routes"]:
        details.append(
            {
                "vehicle": r["vehicle_id"],
                "distance_km": round(r["distance_m"] / 1000.0, 3),
                "load": r["load"],
                "capacity_util_pct": round(min(1.0, r["capacity_ratio"]) * 100.0, 1),
            }
        )
    st.subheader("Vehicle-level breakdown")
    st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)

    opt_export = build_export_df(optimized["routes"], locations)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download optimized routes (CSV)",
            opt_export.to_csv(index=False),
            "optimized_routes.csv",
            "text/csv",
        )
    with c2:
        st.download_button(
            "Download optimized routes (JSON)",
            json.dumps(optimized["routes"], indent=2),
            "optimized_routes.json",
            "application/json",
        )
else:
    st.info("Set fleet and demand assumptions in the sidebar, then click **Run route optimization**.")
    st.subheader("Input preview")
    st.dataframe(working.head(30), use_container_width=True, hide_index=True)
