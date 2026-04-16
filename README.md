# Strategic Vehicle Routing & Fleet Optimization Engine

This project is a **scalable logistics inference engine** for high-velocity route planning decisions. It models delivery design as a **Capacitated Vehicle Routing Problem (CVRP)** and combines optimization with operational analytics so teams can compare baseline dispatch plans against optimized fleet plans in seconds.

## Why this problem matters

Real-world routing is not a trivial shortest-path task. CVRP is **NP-hard**, which means exact search becomes computationally expensive as stops, constraints, and fleet size grow. In production logistics environments, planners still need high-quality routes under strict time pressure.

This engine addresses that by using a solver-driven approach that returns **near-optimal, decision-ready routes quickly**.

## Optimization approach

- **Core model:** CVRP with objective to minimize total travel distance.
- **Solver strategy:** specialized routing optimization via **Google OR-Tools** (constraint programming + local search strategies) to deliver strong solutions in seconds.
- **Scalability posture:** engineered for iterative scenario analysis (fleet size, demand profile, depot choice, and service assumptions) with fast recomputation for dispatcher workflows.

## Business impact

The platform is designed to convert optimization outputs into direct operational outcomes:

- **Minimized total route distance** to reduce route inefficiency.
- **Reduced fuel consumption** through shorter cumulative travel paths.
- **Improved fleet utilization** by balancing route loads against truck capacity.
- **Decision confidence** via side-by-side baseline vs optimized scenario comparison.

## Real-world constraints handled

Current and extensible constraints include:

- **Vehicle capacity limits** (hard capacity per vehicle).
- **Depot location control** (configurable origin node).
- **Customer demand** at each stop.
- **Time-budgeted solving** for operations-friendly latency.

Production extensions typically include:

- **Time windows** (service windows per customer),
- **Driver shift limits** and break rules,
- **Multiple depots** and heterogeneous fleets,
- **Road-network travel times** instead of Euclidean approximations.

## Product interface (Logistics Command Center)

The Streamlit UI is structured for executive and dispatcher use:

- **Map-centric route canvas** with color-coded vehicle paths and directional stop sequence markers.
- **Fleet KPI strip** with total distance, utilization, and estimated CO2 savings.
- **Sidebar logistics controls** for vehicles, depot, capacity, demand multiplier, and solve time.
- **Scenario comparison mode** to benchmark a manual route policy against optimized routing.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data format

Input CSV should include:

- `id` (optional),
- `lat` / `latitude`,
- `lon` / `lng` / `longitude`,
- `demand`.

## Strategic note

This engine is purpose-built for organizations where routing quality directly affects cost, service reliability, and sustainability reporting. It provides a practical bridge between **Operations Research rigor** and **day-to-day dispatch execution**.

