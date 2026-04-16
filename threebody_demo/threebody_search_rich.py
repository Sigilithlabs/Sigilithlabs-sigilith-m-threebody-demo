#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
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


def distance_band(d: float, close_threshold: float, medium_threshold: float) -> str:
    if d < close_threshold:
        return "1"
    if d < medium_threshold:
        return "2"
    return "3"


def encode_state(state: List[Body], close_threshold: float, medium_threshold: float) -> str:
    positions = [(b.x, b.y) for b in state]
    d_ab = pair_distance(positions[0], positions[1])
    d_ac = pair_distance(positions[0], positions[2])
    d_bc = pair_distance(positions[1], positions[2])

    ab = "AB" + distance_band(d_ab, close_threshold, medium_threshold)
    ac = "AC" + distance_band(d_ac, close_threshold, medium_threshold)
    bc = "BC" + distance_band(d_bc, close_threshold, medium_threshold)
    return f"{ab}_{ac}_{bc}"


def simulate(
    initial_state: List[Body],
    dt: float,
    steps: int,
    sample_every: int,
    close_threshold: float,
    medium_threshold: float,
) -> List[str]:
    masses = [b.mass for b in initial_state]
    vec = state_to_vector(initial_state)
    tokens: List[str] = []

    for step in range(steps):
        vec = rk4_step(vec, dt, masses)
        if step % sample_every == 0:
            state = vector_to_state(vec, masses)
            tokens.append(encode_state(state, close_threshold, medium_threshold))

    return tokens


def random_initial_state(rng: random.Random) -> List[Body]:
    positions = []
    velocities = []

    for _ in range(3):
        x = rng.uniform(-0.8, 0.8)
        y = rng.uniform(-0.8, 0.8)
        vx = rng.uniform(-0.7, 0.7)
        vy = rng.uniform(-0.7, 0.7)
        positions.append((x, y))
        velocities.append((vx, vy))

    cx = sum(p[0] for p in positions) / 3.0
    cy = sum(p[1] for p in positions) / 3.0
    cvx = sum(v[0] for v in velocities) / 3.0
    cvy = sum(v[1] for v in velocities) / 3.0

    bodies: List[Body] = []
    for i in range(3):
        x, y = positions[i]
        vx, vy = velocities[i]
        bodies.append(
            Body(
                mass=1.0,
                x=x - cx,
                y=y - cy,
                vx=vx - cvx,
                vy=vy - cvy,
            )
        )
    return bodies


def shannon_entropy(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent


def transition_count(tokens: List[str]) -> int:
    if len(tokens) < 2:
        return 0
    return len(set(zip(tokens[:-1], tokens[1:])))


def richness(tokens: List[str]) -> dict:
    uniq = len(set(tokens))
    ent = shannon_entropy(tokens)
    trans = transition_count(tokens)
    score = 1.0 * uniq + 2.0 * ent + 0.5 * trans

    counts = Counter(tokens)
    top_share = counts.most_common(1)[0][1] / len(tokens) if tokens else 1.0

    return {
        "unique_tokens": uniq,
        "entropy": ent,
        "transition_count": trans,
        "top_token_share": top_share,
        "richness_score": score,
        "counts": counts,
    }


def save_sequence(tokens: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(tokens) + "\n", encoding="utf-8")


def save_initial_state(state: List[Body], path: Path) -> None:
    payload = [
        {
            "mass": b.mass,
            "x": b.x,
            "y": b.y,
            "vx": b.vx,
            "vy": b.vy,
        }
        for b in state
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for the richest symbolic three-body runs.")
    parser.add_argument("--runs", type=int, default=30, help="Number of random runs.")
    parser.add_argument("--top-k", type=int, default=5, help="How many top runs to keep.")
    parser.add_argument("--dt", type=float, default=0.002, help="Integrator timestep.")
    parser.add_argument("--steps", type=int, default=4000, help="Number of steps.")
    parser.add_argument("--sample-every", type=int, default=1, help="Sample every N steps.")
    parser.add_argument("--close-threshold", type=float, default=0.4, help="Distance threshold for band 1.")
    parser.add_argument("--medium-threshold", type=float, default=1.0, help="Distance threshold for band 2.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--outdir", type=str, default="threebody_search", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []

    for run_id in range(args.runs):
        state = random_initial_state(rng)
        tokens = simulate(
            initial_state=state,
            dt=args.dt,
            steps=args.steps,
            sample_every=args.sample_every,
            close_threshold=args.close_threshold,
            medium_threshold=args.medium_threshold,
        )
        stats = richness(tokens)
        results.append(
            {
                "run_id": run_id,
                "state": state,
                "tokens": tokens,
                **stats,
            }
        )

    results.sort(key=lambda x: x["richness_score"], reverse=True)
    top = results[:args.top_k]

    summary = []
    for rank, item in enumerate(top, start=1):
        seq_path = outdir / f"rank_{rank:02d}_run_{item['run_id']:02d}.txt"
        state_path = outdir / f"rank_{rank:02d}_run_{item['run_id']:02d}.json"

        save_sequence(item["tokens"], seq_path)
        save_initial_state(item["state"], state_path)

        summary.append(
            {
                "rank": rank,
                "run_id": item["run_id"],
                "sequence_file": str(seq_path),
                "state_file": str(state_path),
                "unique_tokens": item["unique_tokens"],
                "entropy": round(item["entropy"], 6),
                "transition_count": item["transition_count"],
                "top_token_share": round(item["top_token_share"], 6),
                "richness_score": round(item["richness_score"], 6),
                "top_tokens": item["counts"].most_common(8),
            }
        )

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved top {len(top)} runs to: {outdir}")
    print()
    for row in summary:
        print(
            f"Rank {row['rank']}: run {row['run_id']} | "
            f"unique={row['unique_tokens']} | "
            f"entropy={row['entropy']:.4f} | "
            f"transitions={row['transition_count']} | "
            f"top_share={row['top_token_share']:.4f} | "
            f"richness={row['richness_score']:.2f}"
        )
        print(f"  sequence: {row['sequence_file']}")
        print(f"  state:    {row['state_file']}")
        print(f"  top tokens: {row['top_tokens']}")
        print()


if __name__ == "__main__":
    main()
