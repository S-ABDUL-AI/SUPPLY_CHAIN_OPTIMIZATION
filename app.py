import streamlit as st
import pandas as pd
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import pydeck as pdk
import json
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="🚚 Vehicle Routing Optimization (CVRP)",
    layout="wide",
    page_icon="🚚"
)

# ------------------------------
# App Header
# ------------------------------
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:15px;border-radius:5px;">
        <h1 style="color:#FF4B4B;">🚚 Vehicle Routing Optimization (CVRP)</h1>
        <p>Optimize delivery routes for multiple vehicles considering capacity constraints.</p>
        <p>Cost savings: Less distance = lower fuel and labor costs | Time savings: Deliver faster to customers</p>
        <p> Efficiency: Better use of vehicles | Scalability: Can handle hundreds or thousands of deliveries</p>
        <p><b>Units:</b> Capacity → units, Distance → meters</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data(file_path="routes.csv"):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"File {file_path} not found!")
        return None

uploaded_file = st.file_uploader("Upload your own CSV (id, lat, lon, demand)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Custom CSV loaded")
else:
    data = load_data()
    st.info("Loaded default `routes.csv`")

if data is not None:
    st.subheader("📍 Customer Locations")
    st.dataframe(data)

    # ------------------------------
    # Distance Matrix
    # ------------------------------
    def compute_euclidean_distance_matrix(locations):
        size = len(locations)
        matrix = {}
        for i in range(size):
            matrix[i] = {}
            for j in range(size):
                if i == j:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = int(np.linalg.norm(
                        (locations[i][0] - locations[j][0], locations[i][1] - locations[j][1])
                    ) * 1000)  # meters
        return matrix

    locations = list(zip(data["lat"], data["lon"]))
    demands = data["demand"].tolist()
    depot = 0
    distance_matrix = compute_euclidean_distance_matrix(locations)

    # ------------------------------
    # Sidebar Controls
    # ------------------------------
    st.sidebar.header("🚦 Route Parameters")
    num_vehicles = st.sidebar.number_input("Number of Vehicles", min_value=1, max_value=10, value=3)
    vehicle_capacity = st.sidebar.number_input("Vehicle Capacity (units)", min_value=1, max_value=1000, value=30)
    solver_time = st.sidebar.slider("Solver Time Limit (seconds)", min_value=5, max_value=60, value=10)

    # ------------------------------
    # OR-Tools Model
    # ------------------------------
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [vehicle_capacity] * num_vehicles,
        True,
        "Capacity"
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = solver_time

    solution = routing.SolveWithParameters(search_parameters)

    # ------------------------------
    # Output & Map Visualization
    # ------------------------------
    if solution:
        st.subheader("✅ Optimized Routes Found")
        total_distance = 0
        route_paths = []
        max_load = 0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route_distance = 0
            route_load = 0
            route_coords = []

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += demands[node_index]
                route_coords.append((locations[node_index][0], locations[node_index][1]))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            # Append depot at end
            route_coords.append((locations[depot][0], locations[depot][1]))
            route_paths.append({
                "vehicle_id": vehicle_id + 1,
                "route": route_coords,
                "load": route_load,
                "distance": route_distance
            })
            total_distance += route_distance
            max_load = max(max_load, route_load)

        # ------------------------------
        # KPI Summary Cards
        # ------------------------------
        st.markdown("### 📊 Summary KPIs")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Vehicles", num_vehicles)
        col2.metric("Max Load (units)", max_load)
        col3.metric("Total Distance (meters)", total_distance)

        # Display each route
        for r in route_paths:
            st.markdown(f"**Vehicle {r['vehicle_id']}** | Load: {r['load']} units | Distance: {r['distance']} meters")

        # ------------------------------
        # PyDeck Map
        # ------------------------------
        layers = []
        colors = [[255,0,0], [0,255,0], [0,0,255], [255,165,0], [128,0,128]]
        for i, r in enumerate(route_paths):
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=[{"path": r["route"]}],
                    get_path="path",
                    get_color=colors[i % len(colors)],
                    width_scale=10,
                    width_min_pixels=5,
                    pickable=True,
                )
            )
            for coord in r["route"]:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=[{"position": coord}],
                        get_position="position",
                        get_color=colors[i % len(colors)],
                        get_radius=50,
                    )
                )

        st.subheader("🗺️ Routes Map")
        view_state = pdk.ViewState(
            latitude=np.mean([lat for lat, lon in locations]),
            longitude=np.mean([lon for lat, lon in locations]),
            zoom=11,
            pitch=0
        )
        r = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text":"Vehicle Route"})
        st.pydeck_chart(r)

        # ------------------------------
        # Download Buttons
        # ------------------------------
        routes_export = []
        for r in route_paths:
            for idx, coord in enumerate(r["route"]):
                routes_export.append({
                    "vehicle_id": r["vehicle_id"],
                    "position": idx,
                    "lat": coord[0],
                    "lon": coord[1]
                })
        routes_df = pd.DataFrame(routes_export)
        st.download_button(
            "📥 Download Routes CSV",
            data=routes_df.to_csv(index=False),
            file_name="optimized_routes.csv",
            mime="text/csv"
        )
        st.download_button(
            "📥 Download Routes JSON",
            data=json.dumps(route_paths, indent=4),
            file_name="optimized_routes.json",
            mime="application/json"
        )

    else:
        st.warning("❌ No solution found!")

# ------------------------------
# About the Developer
# ------------------------------
st.sidebar.markdown(
    """
    ---
    **👨‍💻 About the Developer**  
    Sherriff Abdul-Hamid  
    AI Engineer | Data Scientist | Economist  

    **Contact:**  
    [GitHub](https://github.com/S-ABDUL-AI) |  
    [LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/) |  
    📧 Sherriffhamid001@gmail.com
    """
)
