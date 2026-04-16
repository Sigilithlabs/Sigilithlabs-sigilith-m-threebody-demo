# Three-Body Demonstration

This folder contains the reproducible Sigilith-M demonstration for symbolic profiling of planar three-body simulations.

## Files
- `threebody_to_sigilith.py` — simulation-to-symbolic encoding pipeline
- `threebody_search_rich.py` — batch search for structurally rich runs
- `example_sequences/` — representative symbolic trajectories
- `example_profiles/` — Sigilith-M profile outputs

## Pipeline

### 1. Simulate
Planar Newtonian three-body trajectories are generated from randomized balanced initial conditions with fixed seed control.

### 2. Discretize
Each timestep is encoded into pairwise distance-band tokens over AB, AC, and BC, producing a symbolic sequence.

### 3. Rank
Runs are ranked by symbolic richness using:
- unique token count
- Shannon entropy
- distinct transition count

Richness score:

`R = 1.0*U + 2.0*H + 0.5*T`

### 4. Profile
Shortlisted runs are profiled with Sigilith-M structural metrics:
- stability
- repetition ratio
- transition diversity
- drift
- normalized drift
- windowed drift
- stability index
- classification label

### 5. Compare
Representative collapse, constrained, and mobile runs are compared by metric deltas and figure-level summaries.

