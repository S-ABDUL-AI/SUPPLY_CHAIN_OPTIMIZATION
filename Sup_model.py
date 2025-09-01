# model.py
import math
import json
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def haversine_distance(lat1, lon1, lat2, lon2):
    # returns distance in kilometers
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2 * R * math.asin(math.sqrt(a))

def make_distance_matrix(locations_df):
    n = len(locations_df)
    coords = locations_df[['lat', 'lon']].values
    dist_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0
            else:
                dist = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                # convert to meters and round to int for OR-Tools (or keep km*1000)
                dist_matrix[i][j] = int(dist * 1000)
    return dist_matrix

def load_data_from_csv(path):
    df = pd.read_csv(path)
    # required columns: id, lat, lon, demand
    required = {'id', 'lat', 'lon', 'demand'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    # ensure sorted by id
    df = df.sort_values('id').reset_index(drop=True)
    return df

def solve_vrp(locations_df, vehicle_count=3, vehicle_capacity=30, time_limit=30):
    """
    locations_df: pandas DataFrame with columns id, lat, lon, demand
    returns: solution dict with routes, loads, distances (meters) and total distance
    """
    data = {}
    data['locations'] = locations_df[['lat','lon']].values.tolist()
    data['demands'] = locations_df['demand'].astype(int).tolist()
    data['num_locations'] = len(locations_df)
    data['num_vehicles'] = int(vehicle_count)
    data['vehicle_capacities'] = [int(vehicle_capacity)] * data['num_vehicles']
    data['depot'] = 0

    dist_matrix = make_distance_matrix(locations_df)
    data['distance_matrix'] = dist_matrix

    # Create the routing index manager and model.
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        # returns distance between the two nodes
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraint.
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(int(time_limit))

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        return {'status': 'NOT_SOLVED'}

    # Extract routes
    routes = []
    total_distance = 0
    total_load = 0
    capacity_dim = routing.GetDimensionOrDie('Capacity')
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = []
        route_load = 0
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(int(node_index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            route_load += data['demands'][node_index]
        # include end (depot)
        route.append(int(manager.IndexToNode(index)))
        total_distance += route_distance
        total_load += route_load
        routes.append({
            'vehicle_id': vehicle_id,
            'route': route,
            'load': route_load,
            'distance_m': int(route_distance)
        })

    result = {
        'status': 'SUCCESS',
        'routes': routes,
        'total_distance_m': int(total_distance),
        'total_load': int(total_load),
        'vehicle_capacity': vehicle_capacity
    }
    return result

def save_solution(solution_dict, path="solution.json"):
    with open(path, "w") as f:
        json.dump(solution_dict, f, indent=2)
    return path

# Example quick test (only if executed directly)
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "routes.csv"
    df = load_data_from_csv(path)
    sol = solve_vrp(df, vehicle_count=3, vehicle_capacity=30, time_limit=10)
    print(json.dumps(sol, indent=2))
    save_solution(sol, "sample_solution.json")

