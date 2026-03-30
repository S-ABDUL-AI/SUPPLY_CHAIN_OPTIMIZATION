import json
import os

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(
    page_title="Supply chain routing",
    page_icon="🚚",
    layout="wide",
)


def normalize_routes_df(raw: pd.DataFrame) -> pd.DataFrame:
    colmap = {c.lower().strip(): c for c in raw.columns}
    lat_c = next((colmap[k] for k in ("lat", "latitude") if k in colmap), None)
    lon_c = next((colmap[k] for k in ("lon", "lng", "longitude") if k in colmap), None)
    dem_c = colmap.get("demand")
    id_c = colmap.get("id")
    if not all([lat_c, lon_c, dem_c]):
        raise ValueError("CSV needs lat/latitude, lon/longitude, and demand columns.")
    out = pd.DataFrame(
        {
            "id": raw[id_c] if id_c else range(len(raw)),
            "lat": pd.to_numeric(raw[lat_c], errors="coerce"),
            "lon": pd.to_numeric(raw[lon_c], errors="coerce"),
            "demand": pd.to_numeric(raw[dem_c], errors="coerce").fillna(0).astype(int),
        }
    )
    out = out.dropna(subset=["lat", "lon"])
    if out.empty:
        raise ValueError("No valid coordinates after parsing.")
    return out.reset_index(drop=True)


@st.cache_data
def load_default_routes(path: str = "routes.csv"):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def euclidean_meters_matrix(locations):
    n = len(locations)
    mat = {}
    for i in range(n):
        mat[i] = {}
        for j in range(n):
            if i == j:
                mat[i][j] = 0
            else:
                d = np.linalg.norm(
                    np.array(locations[i]) - np.array(locations[j]),
                )
                mat[i][j] = int(d * 111_000)
    return mat


st.title("🚚 Capacitated vehicle routing (CVRP)")
st.caption(
    "Minimize total distance with vehicle capacity constraints — typical last‑mile / depot routing demo. "
    "Distances use planar meters (scaled); swap in a road matrix for production."
)

uploaded = st.file_uploader("Upload stops (CSV): columns id, lat, lon, demand — or use default `routes.csv`", type=["csv"])
if uploaded:
    try:
        data = normalize_routes_df(pd.read_csv(uploaded))
        st.success("Custom file loaded.")
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    raw = load_default_routes()
    if raw is None:
        st.error("Default **routes.csv** not found in the app directory.")
        st.stop()
    try:
        data = normalize_routes_df(raw)
    except Exception as e:
        st.error(str(e))
        st.stop()
    st.info("Using bundled **routes.csv**.")

st.subheader("Stops preview")
st.dataframe(data.head(20), use_container_width=True, hide_index=True)

locations = list(zip(data["lat"], data["lon"]))
demands = data["demand"].tolist()
depot = 0
distance_matrix = euclidean_meters_matrix(locations)

st.sidebar.header("Solver")
num_vehicles = st.sidebar.number_input("Vehicles", min_value=1, max_value=20, value=3)
vehicle_capacity = st.sidebar.number_input("Capacity per vehicle (units)", min_value=1, max_value=10_000, value=30)
solver_time = st.sidebar.slider("Time limit (seconds)", 5, 120, 15)

manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot)
routing = pywrapcp.RoutingModel(manager)


def distance_callback(from_index, to_index):
    return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]


transit_cb = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)


def demand_cb(from_index):
    return demands[manager.IndexToNode(from_index)]


demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
routing.AddDimensionWithVehicleCapacity(
    demand_cb_idx,
    0,
    [vehicle_capacity] * num_vehicles,
    True,
    "Capacity",
)

params = pywrapcp.DefaultRoutingSearchParameters()
params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
params.time_limit.seconds = int(solver_time)

if st.button("Run optimization", type="primary"):
    solution = routing.SolveWithParameters(params)
    if not solution:
        st.warning("No feasible solution — raise capacity or add vehicles.")
    else:
        total_distance = 0
        route_paths = []
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route_distance = 0
            route_load = 0
            route_coords = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route_load += demands[node]
                route_coords.append((locations[node][0], locations[node][1]))
                prev = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(prev, index, vehicle_id)
            route_coords.append((locations[depot][0], locations[depot][1]))
            route_paths.append(
                {
                    "vehicle_id": vehicle_id + 1,
                    "route": route_coords,
                    "load": route_load,
                    "distance_m": route_distance,
                }
            )
            total_distance += route_distance

        m1, m2, m3 = st.columns(3)
        m1.metric("Total distance (m, approx.)", f"{total_distance:,}")
        m2.metric("Vehicles used", num_vehicles)
        m3.metric("Max load on a route", max(r["load"] for r in route_paths))

        for r in route_paths:
            st.write(f"**Vehicle {r['vehicle_id']}** — load **{r['load']}** units · distance **{r['distance_m']:,}** m")

        layers = []
        palette = [[239, 68, 68], [34, 197, 94], [59, 130, 246], [234, 179, 8], [168, 85, 247]]
        for i, r in enumerate(route_paths):
            color = palette[i % len(palette)]
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=[{"path": r["route"]}],
                    get_path="path",
                    get_color=color,
                    width_scale=12,
                    width_min_pixels=4,
                    pickable=True,
                )
            )
        mid_lat = float(np.mean([lat for lat, _ in locations]))
        mid_lon = float(np.mean([lon for _, lon in locations]))
        deck_chart = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=11, pitch=0),
            tooltip={"text": "Route segment"},
        )
        st.subheader("Map")
        st.pydeck_chart(deck_chart)

        export_rows = []
        for r in route_paths:
            for seq, coord in enumerate(r["route"]):
                export_rows.append(
                    {
                        "vehicle_id": r["vehicle_id"],
                        "seq": seq,
                        "lat": coord[0],
                        "lon": coord[1],
                    }
                )
        routes_df = pd.DataFrame(export_rows)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download routes CSV",
                routes_df.to_csv(index=False),
                "optimized_routes.csv",
                "text/csv",
            )
        with c2:
            st.download_button(
                "Download routes JSON",
                json.dumps(route_paths, indent=2),
                "optimized_routes.json",
                "application/json",
            )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Sherriff Abdul-Hamid**  \n"
    "[GitHub](https://github.com/S-ABDUL-AI) · "
    "[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)"
)
