# Strategic Vehicle Routing & Fleet Optimization Engine

This repository delivers a **Scalable Logistics Inference Engine** that turns raw stop-demand data into dispatch-ready plans. It frames routing as a **Capacitated Vehicle Routing Problem (CVRP)** and produces high-quality route allocations fast enough for operational decision cycles.

## Problem class and computational complexity

Fleet routing at scale is not a simple shortest-path lookup. The engine solves CVRP, an **NP-hard combinatorial optimization problem** where exact methods become impractical as stop count, fleet size, and constraints increase.  

Operationally, this means planners need near-optimal solutions under strict latency requirements, not mathematically exact solutions after long runtimes.

## Algorithmic strategy

The optimization stack is built for robust, production-oriented search:

- **Core optimizer:** specialized solver techniques via **Google OR-Tools** (routing model, local search, constraint handling).
- **Metaheuristic posture:** supports and aligns with **metaheuristic search patterns** (e.g., guided local search, simulated annealing-like neighborhood exploration, and genetic-style diversification concepts) to escape local minima and improve solution quality rapidly.
- **Near-optimal in seconds:** designed to return strong feasible routes quickly, then improve within a bounded solve-time budget.
- **Scenario recomputation:** enables rapid reruns across changes in demand, capacity, fleet size, and depot selection.

## Business impact

This engine translates OR outputs into financial and operational gains:

- **Minimizes total route distance** across the active fleet.
- **Reduces fuel consumption** by lowering cumulative travel kilometers.
- **Maximizes fleet utilization** by balancing load assignments against available vehicle capacity.
- **Improves planning confidence** through baseline-vs-optimized scenario comparison.

## Real-world constraints modeled

The system is built to handle practical dispatch constraints:

- **Vehicle capacity** constraints (hard load limits per vehicle).
- **Depot locations** (single depot today, extensible to multi-depot settings).
- **Time windows** for customer service commitments (supported as a production constraint pattern and standard OR extension path).
- **Demand fulfillment** requirements at each customer stop.
- **Solver time limits** to guarantee operations-friendly response times.

## Product capabilities

- **Interactive route map:** color-coded paths with stop sequence markers.
- **Fleet KPI panel:** optimized distance, route delta vs baseline, utilization, and sustainability proxy metrics.
- **What-if controls:** vehicles, capacity, depot selection, demand multiplier, and solve budget.
- **Export artifacts:** route-level CSV/JSON outputs for downstream dispatch execution.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input schema

Input CSV supports:

- `id` (optional),
- `lat` / `latitude`,
- `lon` / `lng` / `longitude`,
- `demand`.

## Strategic positioning

This project is intentionally positioned at the intersection of **Operations Research**, **heuristic search**, and **logistics execution**: a decision engine that prioritizes speed, scalability, and measurable business impact.

