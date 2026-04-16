#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

G = 1.0

@dataclass
class Body:
    mass: float
    x: float
    y: float
    vx: float
    vy: float

def pair_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)

def vec_norm(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)

def center_of_mass(state: List[Body]) -> Tuple[float, float]:
    total_mass = sum(b.mass for b in state)
    cx = sum(b.mass * b.x for b in state) / total_mass
    cy = sum(b.mass * b.y for b in state) / total_mass
    return cx, cy

def state_to_vector(state: List[Body]) -> List[float]:
    vec: List[float] = []
    for b in state:
        vec.extend([b.x, b.y, b.vx, b.vy])
    return vec

def vector_to_state(vec: List[float], masses: List[float]) -> List[Body]:
    bodies: List[Body] = []
    for i in range(3):
        base = 4 * i
        bodies.append(
            Body(
                mass=masses[i],
                x=vec[base],
                y=vec[base + 1],
                vx=vec[base + 2],
                vy=vec[base + 3],
            )
        )
    return bodies

def accelerations(state: List[Body], softening: float = 1e-3) -> List[Tuple[float, float]]:
    accs: List[Tuple[float, float]] = []
    for i, bi in enumerate(state):
        ax = 0.0
        ay = 0.0
        for j, bj in enumerate(state):
            if i == j:
                continue
            dx = bj.x - bi.x
            dy = bj.y - bi.y
            r2 = dx * dx + dy * dy + softening * softening
            r = math.sqrt(r2)
            factor = G * bj.mass / (r2 * r)
            ax += factor * dx
            ay += factor * dy
        accs.append((ax, ay))
    return accs

def derivatives(vec: List[float], masses: List[float]) -> List[float]:
    state = vector_to_state(vec, masses)
    accs = accelerations(state)
    dvec: List[float] = []
    for i, b in enumerate(state):
        ax, ay = accs[i]
        dvec.extend([b.vx, b.vy, ax, ay])
    return dvec

def rk4_step(vec: List[float], dt: float, masses: List[float]) -> List[float]:
    def add_scaled(v1: List[float], v2: List[float], scale: float) -> List[float]:
        return [a + scale * b for a, b in zip(v1, v2)]

    k1 = derivatives(vec, masses)
    k2 = derivatives(add_scaled(vec, k1, dt / 2.0), masses)
    k3 = derivatives(add_scaled(vec, k2, dt / 2.0), masses)
    k4 = derivatives(add_scaled(vec, k3, dt), masses)

    return [
        v + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        for v, a, b, c, d in zip(vec, k1, k2, k3, k4)
    ]

def closest_pair_and_dist(state: List[Body]) -> Tuple[str, float]:
    positions = [(b.x, b.y) for b in state]
    d_ab = pair_distance(positions[0], positions[1])
    d_ac = pair_distance(positions[0], positions[2])
    d_bc = pair_distance(positions[1], positions[2])
    pairs = [("AB", d_ab), ("AC", d_ac), ("BC", d_bc)]
    return min(pairs, key=lambda item: item[1])

def escape_candidate(state: List[Body], radius_threshold: float, speed_threshold: float) -> str | None:
    cx, cy = center_of_mass(state)
    candidates: List[Tuple[str, float]] = []
    labels = ["A", "B", "C"]

    for label, b in zip(labels, state):
        rx = b.x - cx
        ry = b.y - cy
        r = vec_norm(rx, ry)
        v = vec_norm(b.vx, b.vy)
        if r > radius_threshold and v > speed_threshold:
            candidates.append((label, r))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[1], reverse=True)
    return f"ESC_{candidates[0][0]}"

def distance_band(d: float, close_threshold: float, medium_threshold: float) -> str:
    if d < close_threshold:
        return "1"
    if d < medium_threshold:
        return "2"
    return "3"

def encode_state(
    state: List[Body],
    close_threshold: float,
    medium_threshold: float,
    escape_radius: float,
    escape_speed: float,
) -> str:
    esc = escape_candidate(state, escape_radius, escape_speed)
    if esc is not None:
        return esc
    pair, d = closest_pair_and_dist(state)
    band = distance_band(d, close_threshold, medium_threshold)
    return f"{pair}{band}"

def simulate(
    initial_state: List[Body],
    dt: float,
    steps: int,
    sample_every: int,
    close_threshold: float,
    medium_threshold: float,
    escape_radius: float,
    escape_speed: float,
) -> List[str]:
    masses = [b.mass for b in initial_state]
    vec = state_to_vector(initial_state)
    tokens: List[str] = []

    for step in range(steps):
        vec = rk4_step(vec, dt, masses)
        if step % sample_every == 0:
            state = vector_to_state(vec, masses)

            positions = [(b.x, b.y) for b in state]
            d_ab = pair_distance(positions[0], positions[1])
            d_ac = pair_distance(positions[0], positions[2])
            d_bc = pair_distance(positions[1], positions[2])

            ab = "AB" + distance_band(d_ab, close_threshold, medium_threshold)
            ac = "AC" + distance_band(d_ac, close_threshold, medium_threshold)
            bc = "BC" + distance_band(d_bc, close_threshold, medium_threshold)

            token = f"{ab}_{ac}_{bc}"
            tokens.append(token)

    return tokens

def default_initial_state() -> List[Body]:
    return [
        Body(mass=1.0, x=-0.5, y=0.0, vx=0.00, vy=0.55),
        Body(mass=1.0, x=0.5, y=0.0, vx=0.00, vy=-0.55),
        Body(mass=1.0, x=0.0, y=0.4, vx=-0.48, vy=0.00),
    ]

def write_tokens(tokens: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(" ".join(tokens) + "\n", encoding="utf-8")

def print_summary(tokens: List[str]) -> None:
    counts = Counter(tokens)
    print(f"Generated {len(tokens)} tokens")
    print("Top tokens:")
    for token, count in counts.most_common(10):
        print(f"  {token}: {count}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a three-body system and encode it for Sigilith-M.")
    parser.add_argument("--dt", type=float, default=0.01, help="Integrator timestep.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of integration steps.")
    parser.add_argument("--sample-every", type=int, default=10, help="Sample every N steps.")
    parser.add_argument("--close-threshold", type=float, default=0.8, help="Closest-pair threshold for band 1.")
    parser.add_argument("--medium-threshold", type=float, default=1.8, help="Closest-pair threshold for band 2.")
    parser.add_argument("--escape-radius", type=float, default=4.0, help="COM-relative radius threshold for escape flag.")
    parser.add_argument("--escape-speed", type=float, default=0.9, help="Speed threshold for escape flag.")
    parser.add_argument("--output", type=str, default="threebody.txt", help="Output token file.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    initial = default_initial_state()
    tokens = simulate(
        initial_state=initial,
        dt=args.dt,
        steps=args.steps,
        sample_every=args.sample_every,
        close_threshold=args.close_threshold,
        medium_threshold=args.medium_threshold,
        escape_radius=args.escape_radius,
        escape_speed=args.escape_speed,
    )

    output_path = Path(args.output)
    write_tokens(tokens, output_path)

    print(f"Saved token sequence to: {output_path}")
    print_summary(tokens)

if __name__ == "__main__":
    main()
