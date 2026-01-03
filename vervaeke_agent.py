#!/usr/bin/env python3
"""
Vervaeke Agent v1.3.1 (Persistence + Multi-Agent + Metrics + CLI)
"Belief-Guided Sandpile (Insight Agent)"

Key features:
- Frozen EnvData (immutable perception boundary).
- Full Persistence: Save/Load simulation state (including RNG) to JSON.
- Multi-Agent: Run simulations with 1 to N agents.
- Metrics & Visualization: Track energy, tension, and beliefs over time.
- CLI: Extensive command-line arguments for batch runs.

Fixes added:
- Safe spawns: agents spawn ONLY on EMPTY tiles (not FOOD/HAZARD/MARKERS/HOME).
- Optional lean checkpoints: you can exclude metrics from saves to avoid gigantic JSON.
  On load, missing trackers are re-created automatically so sim still runs.

Usage:
  python3 vervaeke_agent.py --help
  python3 vervaeke_agent.py --agents 3 --steps 1000 --visualize
  python3 vervaeke_agent.py --agents 3 --steps 1000 --save state.json --no-save-metrics
"""

from __future__ import annotations

import argparse
import random
import sys
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================
# Helpers: serialize Python's RNG state (tuples) to JSON-friendly lists
# ============================================================

def _tuple_to_list(obj):
    if isinstance(obj, tuple):
        return [_tuple_to_list(x) for x in obj]
    if isinstance(obj, list):
        return [_tuple_to_list(x) for x in obj]
    return obj


def _list_to_tuple(obj):
    if isinstance(obj, list):
        return tuple(_list_to_tuple(x) for x in obj)
    return obj


# ============================================================
# Layer 1: Ontological Primitives
# ============================================================

class Direction(str, Enum):
    """
    Canonical index order:
      0=NORTH, 1=SOUTH, 2=WEST, 3=EAST, 4=STAY (only if include_stay=True)
    """
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    WEST = "WEST"
    EAST = "EAST"
    STAY = "STAY"

    @classmethod
    def ordered_moves(cls, include_stay: bool) -> List["Direction"]:
        moves: List[Direction] = [cls.NORTH, cls.SOUTH, cls.WEST, cls.EAST]
        if include_stay:
            moves.append(cls.STAY)
        return moves

    @classmethod
    def from_index(cls, idx: int, include_stay: bool) -> "Direction":
        moves = cls.ordered_moves(include_stay)
        if idx < 0 or idx >= len(moves):
            raise ValueError(
                f"Invalid action index {idx} for include_stay={include_stay} (valid 0..{len(moves)-1})"
            )
        return moves[idx]

    def to_vector(self) -> Tuple[int, int]:
        vectors: Dict[Direction, Tuple[int, int]] = {
            Direction.NORTH: (0, -1),
            Direction.SOUTH: (0, 1),
            Direction.WEST: (-1, 0),
            Direction.EAST: (1, 0),
            Direction.STAY: (0, 0),
        }
        return vectors[self]


class CellType(str, Enum):
    EMPTY = "empty"
    FOOD = "food"
    HAZARD = "hazard"
    SAFE_MARKER = "safe_marker"
    DANGER_MARKER = "danger_marker"
    HOME_MARKER = "home_marker"


# ============================================================
# Layer 2: Perception Data (Immutable boundary object)
# ============================================================

class EnvData(BaseModel):
    """
    Immutable snapshot of what the agent sees.
    Frozen so the agent can't mutate the input boundary.
    """
    model_config = ConfigDict(frozen=True)

    food_dist: int = Field(ge=0)
    food_dir: int = Field(ge=0, le=3)

    hazard_dist: int = Field(ge=0)
    hazard_dir: int = Field(ge=0, le=3)

    safe_marker_dist: int = Field(ge=0)
    safe_dir: int = Field(ge=0, le=3)

    danger_marker_dist: int = Field(ge=0)
    danger_dir: int = Field(ge=0, le=3)

    home_marker_dist: int = Field(ge=0)
    home_dir: int = Field(ge=0, le=3)


# ============================================================
# Layer 3: Belief System (Bayesian + Sandpile hooks)
# ============================================================

class Belief(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    description: str
    samples: int = Field(default=0, ge=0)
    successes: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sanity(self) -> "Belief":
        if self.successes > self.samples:
            raise ValueError("successes cannot exceed samples")
        return self

    def verify(self, success: bool) -> None:
        self.samples += 1
        if success:
            self.successes += 1
        # Laplace smoothing: avoids premature certainty
        self.confidence = (self.successes + 1) / (self.samples + 2)

    def shatter(self) -> None:
        # Paradigm shift: push confidence down, but keep inertia
        self.samples += 5
        self.confidence = 0.2

    def __str__(self) -> str:
        return f"[{self.confidence * 100:.1f}%] {self.description}"


class BeliefEngine(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    beliefs: Dict[str, Belief] = Field(default_factory=dict)

    def init_priors_if_empty(self) -> None:
        if self.beliefs:
            return
        self.beliefs["FOOD_APPROACH"] = Belief(description="Approaching FOOD yields Energy")
        self.beliefs["SAFE_MARKER_TOWARD"] = Belief(description="Following SAFE markers yields good outcomes")
        self.beliefs["DANGER_MARKER_AWAY"] = Belief(description="Avoiding DANGER markers preserves Integrity")

    def get_belief(self, key: str) -> Belief:
        if key not in self.beliefs:
            self.beliefs[key] = Belief(description=f"Hypothesis: {key}")
        return self.beliefs[key]

    def get_confidence(self, key: str) -> float:
        return self.beliefs[key].confidence if key in self.beliefs else 0.5  # maximum-entropy prior

    def force_paradigm_shift(self, key: str) -> None:
        if key in self.beliefs:
            self.beliefs[key].shatter()

    def form_hypothesis(self, env: EnvData, action_idx: int) -> Optional[Tuple[str, Belief]]:
        # Only map directional moves into hypotheses (ignore STAY)
        if action_idx < 0 or action_idx > 3:
            return None

        if env.food_dist < 3 and action_idx == env.food_dir:
            key = "FOOD_APPROACH"
            return (key, self.get_belief(key))

        if env.safe_marker_dist < 3 and action_idx == env.safe_dir:
            key = "SAFE_MARKER_TOWARD"
            return (key, self.get_belief(key))

        if env.danger_marker_dist < 3:
            if action_idx != env.danger_dir:
                key = "DANGER_MARKER_AWAY"
                return (key, self.get_belief(key))
            key = "DANGER_APPROACH"
            return (key, self.get_belief(key))

        return None


# ============================================================
# Layer 4: Vitality (Physics state)
# ============================================================

class Vitality(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    energy: float = Field(default=80.0, ge=0.0)
    integrity: float = Field(default=100.0, ge=0.0)
    max_energy: float | None = Field(default=100.0, gt=0.0)

    def _cap_energy(self, value: float) -> float:
        value = max(0.0, value)
        if self.max_energy is not None:
            value = min(self.max_energy, value)
        return value

    def add_energy(self, delta: float) -> None:
        self.energy = self._cap_energy(self.energy + delta)

    def add_integrity(self, delta: float) -> None:
        self.integrity = max(0.0, self.integrity + delta)

    @property
    def is_dead(self) -> bool:
        return self.energy <= 0.0 or self.integrity <= 0.0


# ============================================================
# Layer 5: Metrics Tracking
# ============================================================

class AgentSnapshot(BaseModel):
    tick: int
    energy: float
    integrity: float
    tension: float
    in_confusion: bool
    x: int
    y: int
    direction: str
    belief_food: float
    belief_safe: float
    belief_danger: float


class MetricsTracker(BaseModel):
    model_config = ConfigDict(validate_assignment=False)

    agent_id: str
    snapshots: List[AgentSnapshot] = Field(default_factory=list)

    def record(self, agent: "VervaekeAgent", tick: int, x: int, y: int) -> None:
        self.snapshots.append(
            AgentSnapshot(
                tick=tick,
                energy=agent.vitality.energy,
                integrity=agent.vitality.integrity,
                tension=agent.tension,
                in_confusion=agent.in_confusion_state,
                x=x,
                y=y,
                direction=agent.current_direction.value,
                belief_food=agent.belief_engine.get_confidence("FOOD_APPROACH"),
                belief_safe=agent.belief_engine.get_confidence("SAFE_MARKER_TOWARD"),
                belief_danger=agent.belief_engine.get_confidence("DANGER_MARKER_AWAY"),
            )
        )

    def get_trajectory(self) -> List[Tuple[int, int]]:
        return [(s.x, s.y) for s in self.snapshots]

    def get_energy_series(self) -> Tuple[List[int], List[float]]:
        ticks = [s.tick for s in self.snapshots]
        energies = [s.energy for s in self.snapshots]
        return ticks, energies

    def get_belief_series(self, belief_key: str) -> Tuple[List[int], List[float]]:
        ticks = [s.tick for s in self.snapshots]
        if belief_key == "FOOD_APPROACH":
            vals = [s.belief_food for s in self.snapshots]
        elif belief_key == "SAFE_MARKER_TOWARD":
            vals = [s.belief_safe for s in self.snapshots]
        elif belief_key == "DANGER_MARKER_AWAY":
            vals = [s.belief_danger for s in self.snapshots]
        else:
            vals = [0.5 for _ in ticks]
        return ticks, vals


# ============================================================
# Layer 6: The Agent (Actor)
# ============================================================

class VervaekeAgent(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    agent_id: str = Field(default="agent_0")
    vitality: Vitality = Field(default_factory=Vitality)
    belief_engine: BeliefEngine = Field(default_factory=BeliefEngine)

    include_stay: bool = Field(default=True)

    current_direction: Direction = Field(default=Direction.STAY)
    active_hypothesis_key: str | None = None

    # SOC (sandpile)
    tension: float = Field(default=0.0, ge=0.0)
    tension_threshold: float = Field(default=25.0, gt=0.0)
    in_confusion_state: bool = False

    tick: int = Field(default=0, ge=0)

    def model_post_init(self, __context: object) -> None:
        self.belief_engine.init_priors_if_empty()

    def calculate_salience(self, env: EnvData) -> List[float]:
        n_actions = 5 if self.include_stay else 4

        if self.in_confusion_state:
            return [random.random() for _ in range(n_actions)]

        scores = [0.0 for _ in range(n_actions)]
        urgency = max(0.0, (80.0 - self.vitality.energy) / 80.0)

        conf_food = self.belief_engine.get_confidence("FOOD_APPROACH")
        conf_safe = self.belief_engine.get_confidence("SAFE_MARKER_TOWARD")
        conf_danger = self.belief_engine.get_confidence("DANGER_MARKER_AWAY")

        w_food = (1.0 + urgency * 5.0) * (conf_food * 2.0)
        w_safe = (0.5 + urgency * 2.0) * (conf_safe * 2.0)
        w_danger = (-3.0 - urgency * 2.0) * (conf_danger * 2.0)
        w_home = (urgency * 10.0) if self.vitality.energy < 30 else 0.1

        if env.food_dist < 5:
            scores[env.food_dir] += w_food / (env.food_dist + 1)
        if env.hazard_dist < 3:
            scores[env.hazard_dir] += w_danger / (env.hazard_dist + 1)
        if env.safe_marker_dist < 4:
            scores[env.safe_dir] += w_safe / (env.safe_marker_dist + 1)
        if env.danger_marker_dist < 3:
            scores[env.danger_dir] += w_danger / (env.danger_marker_dist + 1)
        if env.home_marker_dist < 15:
            scores[env.home_dir] += w_home / (env.home_marker_dist + 1)

        if self.include_stay:
            stay_idx = 4
            if self.vitality.energy < 15:
                scores[stay_idx] += 0.25 + urgency * 0.5
            else:
                scores[stay_idx] += 0.01

        return scores

    def decide_action(self, env: EnvData) -> int:
        scores = self.calculate_salience(env)
        best_idx = max(range(len(scores)), key=lambda i: scores[i] + i * 0.001)

        hyp = self.belief_engine.form_hypothesis(env, best_idx)
        self.active_hypothesis_key = hyp[0] if hyp else None
        self.current_direction = Direction.from_index(best_idx, include_stay=self.include_stay)
        return best_idx

    def verify_hypothesis(self, outcome_delta: float) -> None:
        prediction_error = 0.0

        if self.active_hypothesis_key:
            belief = self.belief_engine.get_belief(self.active_hypothesis_key)

            expected_gain = belief.confidence > 0.5
            actual_gain = outcome_delta >= 0.0

            if expected_gain and not actual_gain:
                prediction_error = abs(outcome_delta) * 2.0
            elif (not expected_gain) and actual_gain:
                prediction_error = 10.0

            desc = belief.description
            if "FOOD" in desc:
                belief.verify(outcome_delta > 0)
            elif "SAFE" in desc:
                belief.verify(outcome_delta >= 0)
            elif "DANGER" in desc:
                belief.verify(outcome_delta >= 0)
            else:
                belief.verify(outcome_delta >= 0)
        else:
            if outcome_delta < -5:
                prediction_error = 10.0

        self.tension = max(0.0, self.tension + prediction_error - 2.0)

        if self.tension > self.tension_threshold:
            self.trigger_avalanche()
        elif self.in_confusion_state and prediction_error == 0.0 and outcome_delta >= 0.0:
            self.in_confusion_state = False
            self.tension = 0.0

    def trigger_avalanche(self) -> None:
        print(f"⚡ [{self.agent_id}] AVALANCHE at tick {self.tick}: Frame breaking event ⚡")
        if self.active_hypothesis_key:
            self.belief_engine.force_paradigm_shift(self.active_hypothesis_key)
        self.in_confusion_state = True
        self.tension = 0.0


# ============================================================
# Layer 7: Environment (Headless Grid) - Multi-Agent + Persistence
# ============================================================

class Pos(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class HeadlessEnv(BaseModel):
    """
    validate_assignment=False for performance (grid ops are frequent).
    """
    model_config = ConfigDict(validate_assignment=False)

    width: int = Field(default=10, ge=5)
    height: int = Field(default=10, ge=5)
    grid: List[List[CellType]] = Field(default_factory=list)
    agent_positions: Dict[str, Pos] = Field(default_factory=dict)
    betrayal_mode: bool = False

    # Content knobs (persisted)
    food_count: int = Field(default=8, ge=0)
    hazard_count: int = Field(default=4, ge=0)
    safe_count: int = Field(default=6, ge=0)
    danger_count: int = Field(default=3, ge=0)
    home_at_center: bool = True

    def model_post_init(self, __context: object) -> None:
        if not self.grid:
            self.grid = [[CellType.EMPTY for _ in range(self.width)] for _ in range(self.height)]
            if self.home_at_center:
                cx, cy = self.width // 2, self.height // 2
                self.grid[cy][cx] = CellType.HOME_MARKER
            self._spawn_random_objects()

    def _is_occupied(self, x: int, y: int) -> bool:
        return any(p.x == x and p.y == y for p in self.agent_positions.values())

    def _is_spawnable(self, x: int, y: int) -> bool:
        # Fix: only spawn on EMPTY tiles, and never on other agents
        if self._is_occupied(x, y):
            return False
        return self.grid[y][x] == CellType.EMPTY

    def _find_random_empty_cell(self, tries: int = 500) -> Optional[Tuple[int, int]]:
        for _ in range(tries):
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            if self._is_spawnable(x, y):
                return (x, y)

        # Fallback: scan the whole grid (deterministic-ish and guaranteed if any empty exists)
        for y in range(self.height):
            for x in range(self.width):
                if self._is_spawnable(x, y):
                    return (x, y)
        return None

    def _place_random(self, cell_type: CellType) -> None:
        for _ in range(500):
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            if self._is_occupied(x, y):
                continue
            if self.grid[y][x] == CellType.EMPTY:
                self.grid[y][x] = cell_type
                return

    def _spawn_random_objects(self) -> None:
        for _ in range(self.food_count):
            self._place_random(CellType.FOOD)
        for _ in range(self.hazard_count):
            self._place_random(CellType.HAZARD)
        for _ in range(self.safe_count):
            self._place_random(CellType.SAFE_MARKER)
        for _ in range(self.danger_count):
            self._place_random(CellType.DANGER_MARKER)

    def add_agent(self, agent_id: str, x: int | None = None, y: int | None = None) -> None:
        """
        Fix vs earlier drafts:
        - x=0/y=0 are valid positions; only None means "unspecified".
        - Spawn must be on EMPTY tiles only (no FOOD/HAZARD/MARKERS/HOME).
        """
        if x is None or y is None:
            found = self._find_random_empty_cell()
            if found is None:
                raise RuntimeError("No EMPTY tiles available to spawn an agent.")
            x, y = found
        else:
            # If explicit coordinates are provided, enforce spawn rules.
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"Spawn out of bounds: ({x},{y}) for grid {self.width}x{self.height}")
            if not self._is_spawnable(x, y):
                raise ValueError(f"Spawn cell not EMPTY or occupied at ({x},{y}) (cell={self.grid[y][x]})")

        self.agent_positions[agent_id] = Pos(x=x, y=y)

    def compute_next_pos(self, pos: Pos, direction: Direction) -> Pos:
        dx, dy = direction.to_vector()
        nx = max(0, min(self.width - 1, pos.x + dx))
        ny = max(0, min(self.height - 1, pos.y + dy))
        return Pos(x=nx, y=ny)

    def _nearest(self, from_pos: Pos, predicate: Callable[[CellType], bool]) -> Tuple[int, int]:
        best_dist = 99
        best_dir = 0
        ax, ay = from_pos.x, from_pos.y

        def calc_dir(tx: int, ty: int) -> int:
            if ty < ay:
                return 0  # NORTH
            if ty > ay:
                return 1  # SOUTH
            if tx < ax:
                return 2  # WEST
            if tx > ax:
                return 3  # EAST
            return 0

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if not predicate(cell):
                    continue
                dist = abs(x - ax) + abs(y - ay)
                if dist < best_dist:
                    best_dist = dist
                    best_dir = calc_dir(x, y)

        return best_dist, best_dir

    def get_env_data(self, agent_id: str) -> EnvData:
        a = self.agent_positions[agent_id]

        food_dist, food_dir = self._nearest(a, lambda c: c == CellType.FOOD)
        hazard_dist, hazard_dir = self._nearest(a, lambda c: c == CellType.HAZARD)
        safe_dist, safe_dir = self._nearest(a, lambda c: c == CellType.SAFE_MARKER)
        danger_dist, danger_dir = self._nearest(a, lambda c: c == CellType.DANGER_MARKER)
        home_dist, home_dir = self._nearest(a, lambda c: c == CellType.HOME_MARKER)

        return EnvData(
            food_dist=food_dist, food_dir=food_dir,
            hazard_dist=hazard_dist, hazard_dir=hazard_dir,
            safe_marker_dist=safe_dist, safe_dir=safe_dir,
            danger_marker_dist=danger_dist, danger_dir=danger_dir,
            home_marker_dist=home_dist, home_dir=home_dir,
        )

    def apply_cell_effects(self, agent: VervaekeAgent, pos: Pos, chosen_dir: Direction) -> float:
        """
        Returns outcome_delta for this tick for this agent.
        """
        outcome_delta = 0.0
        cell = self.grid[pos.y][pos.x]

        if cell == CellType.FOOD:
            agent.vitality.add_energy(20.0)
            outcome_delta += 20.0
            self.grid[pos.y][pos.x] = CellType.EMPTY
            self._place_random(CellType.FOOD)

        elif cell == CellType.HAZARD:
            agent.vitality.add_energy(-15.0)
            agent.vitality.add_integrity(-5.0)
            outcome_delta -= 15.0

        elif cell == CellType.SAFE_MARKER:
            if self.betrayal_mode:
                agent.vitality.add_energy(-10.0)
                outcome_delta -= 10.0

        elif cell == CellType.DANGER_MARKER:
            agent.vitality.add_energy(-5.0)
            outcome_delta -= 5.0

        # Recovery if STAY is enabled and chosen
        if agent.include_stay and chosen_dir == Direction.STAY:
            agent.vitality.add_energy(+1.0)
            outcome_delta += 1.0

        # Baseline decay (metabolism)
        agent.vitality.add_energy(-0.5)
        outcome_delta -= 0.5

        return outcome_delta


# ============================================================
# Layer 8: Full-state Persistence
# ============================================================

class SimulationState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    agents: Dict[str, VervaekeAgent] = Field(default_factory=dict)
    env: HeadlessEnv
    tick: int = Field(default=0, ge=0)
    metrics: Dict[str, MetricsTracker] = Field(default_factory=dict)

    # Simulation knobs that affect dynamics (persisted)
    betrayal_step: int = Field(default=200, ge=1)
    collision_policy: Literal["allow", "block"] = "allow"
    shuffle_order: bool = True
    metrics_every: int = Field(default=1, ge=0)  # 0 disables metrics recording entirely

    # Fix: allow checkpoints to omit metrics to keep JSON small
    persist_metrics: bool = True

    # RNG persistence (JSON-friendly)
    rng_state: list | None = None

    def ensure_trackers(self) -> None:
        """
        If we load a checkpoint without metrics (or missing some agents),
        rebuild trackers so the simulation can continue cleanly.
        """
        for aid in self.agents.keys():
            if aid not in self.metrics:
                self.metrics[aid] = MetricsTracker(agent_id=aid)

    def capture_rng(self) -> None:
        self.rng_state = _tuple_to_list(random.getstate())

    def restore_rng(self) -> None:
        if self.rng_state is not None:
            random.setstate(_list_to_tuple(self.rng_state))

    def save_to_file(self, path: str, *, include_metrics: Optional[bool] = None) -> None:
        """
        include_metrics:
          - None: follow self.persist_metrics
          - True/False: override for this save call
        """
        self.capture_rng()
        if include_metrics is None:
            include_metrics = self.persist_metrics

        exclude = set()
        if not include_metrics:
            exclude.add("metrics")

        Path(path).write_text(
            self.model_dump_json(indent=2, exclude=exclude),
            encoding="utf-8",
        )

    @classmethod
    def load_from_file(cls, path: str) -> "SimulationState":
        state = cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
        state.restore_rng()
        state.ensure_trackers()
        return state


# ============================================================
# Visualization
# ============================================================

def visualize_simulation(state: SimulationState, output_dir: str = "plots") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Install with: pip install matplotlib")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Trajectories
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, state.env.width - 0.5)
    ax.set_ylim(-0.5, state.env.height - 0.5)
    ax.set_aspect("equal")
    ax.set_title("Agent Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)

    any_traj = False
    for agent_id, tracker in state.metrics.items():
        traj = tracker.get_trajectory()
        if not traj:
            continue
        any_traj = True
        xs, ys = zip(*traj)
        ax.plot(xs, ys, marker="o", markersize=2, linewidth=1, alpha=0.7, label=agent_id)
        ax.plot(xs[0], ys[0], marker="o", markersize=8, linewidth=0)   # start
        ax.plot(xs[-1], ys[-1], marker="s", markersize=8, linewidth=0)  # end

    if any_traj:
        ax.legend()

    fig.savefig(Path(output_dir) / "trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Energy
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Energy Over Time")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)

    any_energy = False
    for agent_id, tracker in state.metrics.items():
        ticks, energies = tracker.get_energy_series()
        if ticks:
            any_energy = True
            ax.plot(ticks, energies, linewidth=1.5, label=agent_id)

    if any_energy:
        ax.legend()

    fig.savefig(Path(output_dir) / "energy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Beliefs (3 panels)
    belief_keys = ["FOOD_APPROACH", "SAFE_MARKER_TOWARD", "DANGER_MARKER_AWAY"]
    titles = ["Belief: Food Approach", "Belief: Safe Markers", "Belief: Danger Avoidance"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, key, title in zip(axes, belief_keys, titles):
        ax.set_title(title)
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, linestyle="--", alpha=0.3)

        any_line = False
        for agent_id, tracker in state.metrics.items():
            ticks, vals = tracker.get_belief_series(key)
            if ticks:
                any_line = True
                ax.plot(ticks, vals, linewidth=1.5, label=agent_id)

        if any_line:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Tick")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "beliefs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Confusion states
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_title("Confusion States Over Time")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Confusion")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    any_conf = False
    for agent_id, tracker in state.metrics.items():
        ticks = [s.tick for s in tracker.snapshots]
        vals = [1 if s.in_confusion else 0 for s in tracker.snapshots]
        if ticks:
            any_conf = True
            ax.step(ticks, vals, where="post", linewidth=1.5, label=agent_id)

    if any_conf:
        ax.legend()

    fig.savefig(Path(output_dir) / "confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plots to: {output_dir}/")


# ============================================================
# Simulation Controller
# ============================================================

def make_initial_state(
    *,
    num_agents: int = 1,
    include_stay: bool = True,
    max_energy: float | None = 100.0,
    width: int = 10,
    height: int = 10,
    betrayal_step: int = 200,
    collision_policy: Literal["allow", "block"] = "allow",
    shuffle_order: bool = True,
    metrics_every: int = 1,
    persist_metrics: bool = True,
    seed: int | None = None,
    food: int = 8,
    hazard: int = 4,
    safe: int = 6,
    danger: int = 3,
    home_center: bool = True,
) -> SimulationState:
    if seed is not None:
        random.seed(seed)

    env = HeadlessEnv(
        width=width,
        height=height,
        food_count=food,
        hazard_count=hazard,
        safe_count=safe,
        danger_count=danger,
        home_at_center=home_center,
    )

    agents: Dict[str, VervaekeAgent] = {}
    metrics: Dict[str, MetricsTracker] = {}

    for i in range(num_agents):
        agent_id = f"agent_{i}"
        agents[agent_id] = VervaekeAgent(
            agent_id=agent_id,
            include_stay=include_stay,
            vitality=Vitality(max_energy=max_energy),
        )
        metrics[agent_id] = MetricsTracker(agent_id=agent_id)
        env.add_agent(agent_id)

    state = SimulationState(
        agents=agents,
        env=env,
        tick=0,
        metrics=metrics,
        betrayal_step=betrayal_step,
        collision_policy=collision_policy,
        shuffle_order=shuffle_order,
        metrics_every=metrics_every,
        persist_metrics=persist_metrics,
    )
    state.capture_rng()
    return state


def _resolve_moves_block(desired: Dict[str, Pos]) -> Dict[str, bool]:
    """
    Returns dict agent_id -> allowed_to_move.
    Block rule: if multiple agents desire the same cell, none of them move.
    """
    buckets: Dict[Tuple[int, int], List[str]] = {}
    for aid, pos in desired.items():
        buckets.setdefault((pos.x, pos.y), []).append(aid)

    allowed: Dict[str, bool] = {}
    for _, aids in buckets.items():
        if len(aids) == 1:
            allowed[aids[0]] = True
        else:
            for aid in aids:
                allowed[aid] = False
    return allowed


def step_once(state: SimulationState) -> None:
    """
    One tick for all agents:
      perceive -> decide -> (simultaneous move resolve) -> act -> learn -> record metrics
    """
    state.tick += 1

    # Betrayal trigger (persisted env.betrayal_mode is source of truth)
    if (state.tick == state.betrayal_step) and (not state.env.betrayal_mode):
        print("\n--- BETRAYAL PHASE: Safe markers start hurting ---\n")
        state.env.betrayal_mode = True

    agent_ids = list(state.agents.keys())
    if state.shuffle_order:
        random.shuffle(agent_ids)

    # 1) Perceive + decide
    actions: Dict[str, int] = {}
    chosen_dirs: Dict[str, Direction] = {}

    for aid in agent_ids:
        agent = state.agents[aid]
        if agent.vitality.is_dead:
            continue
        agent.tick = state.tick

        env_data = state.env.get_env_data(aid)
        action_idx = agent.decide_action(env_data)

        actions[aid] = action_idx
        chosen_dirs[aid] = agent.current_direction

    # 2) Compute desired positions
    desired_pos: Dict[str, Pos] = {}
    for aid, action_idx in actions.items():
        agent = state.agents[aid]
        cur = state.env.agent_positions[aid]
        direction = Direction.from_index(action_idx, include_stay=agent.include_stay)
        desired_pos[aid] = state.env.compute_next_pos(cur, direction)

    # 3) Resolve collisions (optional)
    if state.collision_policy == "block":
        allowed = _resolve_moves_block(desired_pos)
        for aid, ok in allowed.items():
            if not ok:
                desired_pos[aid] = state.env.agent_positions[aid]  # revert to current

    # Apply moves
    for aid, pos in desired_pos.items():
        state.env.agent_positions[aid] = pos

    # 4) Interactions + metabolism; deterministic ordering for shared cells
    outcome: Dict[str, float] = {}
    for aid in sorted(desired_pos.keys()):
        agent = state.agents[aid]
        if agent.vitality.is_dead:
            continue
        pos = state.env.agent_positions[aid]
        delta = state.env.apply_cell_effects(agent, pos, chosen_dirs[aid])
        outcome[aid] = delta

    # 5) Learning (belief updates, SOC)
    for aid, delta in outcome.items():
        agent = state.agents[aid]
        agent.verify_hypothesis(delta)

    # 6) Metrics
    if state.metrics_every > 0 and (state.tick % state.metrics_every == 0):
        state.ensure_trackers()
        for aid, agent in state.agents.items():
            if agent.vitality.is_dead:
                continue
            pos = state.env.agent_positions[aid]
            state.metrics[aid].record(agent, state.tick, pos.x, pos.y)


def run_simulation(
    state: SimulationState,
    *,
    total_steps: int,
    report_every: int = 25,
    save_path: str | None = None,
    save_every: int | None = None,
    quiet: bool = False,
) -> SimulationState:
    if not quiet:
        print("=== Vervaeke Agent v1.3.1 ===")
        print(f"Steps: {total_steps} | Betrayal at: {state.betrayal_step}")
        print(f"Agents: {len(state.agents)} | Grid: {state.env.width}x{state.env.height}")
        print(f"Collision: {state.collision_policy} | Shuffle order: {state.shuffle_order}")
        print(f"Metrics: every {state.metrics_every} ticks (0 disables)")
        print(f"Persist metrics in saves: {state.persist_metrics}")
        if save_path:
            print(f"Save target: {save_path} (save_every={save_every})")
        print()

    for _ in range(total_steps):
        step_once(state)

        if (not quiet) and report_every and (state.tick % report_every == 0 or state.tick == state.betrayal_step):
            print(f"\n=== Tick {state.tick} ===")
            for aid, agent in state.agents.items():
                pos = state.env.agent_positions.get(aid)
                if pos is None:
                    continue

                status = "DEAD" if agent.vitality.is_dead else "ALIVE"
                confusion = "YES" if agent.in_confusion_state else "no"
                print(
                    f"[{aid}] {status} Pos=({pos.x},{pos.y}) Dir={agent.current_direction.value:>5} "
                    f"E={agent.vitality.energy:5.1f} I={agent.vitality.integrity:5.1f} "
                    f"Tension={agent.tension:5.1f} Confusion={confusion}"
                )

        if save_path and save_every and (state.tick % save_every == 0):
            state.save_to_file(save_path)

        if all(agent.vitality.is_dead for agent in state.agents.values()):
            if not quiet:
                print(f"\nAll agents died at tick {state.tick}")
            break

    if save_path:
        state.save_to_file(save_path)
        if not quiet:
            print(f"\nSaved final state to: {save_path}")

    return state


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vervaeke Agent Simulation v1.3.1 (persistent, multi-agent, metrics, visualization)"
    )

    # Control
    parser.add_argument("--steps", type=int, default=500, help="Steps to run this session (default: 500)")
    parser.add_argument("--betrayal", type=int, default=200, help="Tick where betrayal starts (default: 200)")
    parser.add_argument("--agents", type=int, default=1, help="Number of agents (new run only)")
    parser.add_argument("--width", type=int, default=10, help="Grid width (new run only)")
    parser.add_argument("--height", type=int, default=10, help="Grid height (new run only)")
    parser.add_argument("--max-energy", type=float, default=100.0, help="Energy cap (default: 100)")
    parser.add_argument("--no-stay", action="store_true", help="Disable STAY action")

    # World content knobs (new run only)
    parser.add_argument("--food", type=int, default=8)
    parser.add_argument("--hazard", type=int, default=4)
    parser.add_argument("--safe", type=int, default=6)
    parser.add_argument("--danger", type=int, default=3)
    parser.add_argument("--no-home-center", action="store_true", help="Do not place HOME_MARKER at center")

    # Dynamics knobs
    parser.add_argument("--collision", choices=["allow", "block"], default="allow")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable per-tick agent-order shuffling")
    parser.add_argument("--metrics-every", type=int, default=1, help="Record metrics every N ticks (0 disables)")

    # I/O
    parser.add_argument("--seed", type=int, help="Random seed (new run only)")
    parser.add_argument("--load", type=str, help="Load state from JSON and continue")
    parser.add_argument("--save", type=str, help="Save state to JSON")
    parser.add_argument("--save-every", type=int, help="Checkpoint every N ticks")

    # Fix: keep checkpoint files from exploding
    parser.add_argument(
        "--no-save-metrics",
        action="store_true",
        help="Exclude metrics from saved JSON (smaller checkpoints; plotting requires metrics)",
    )

    # Output
    parser.add_argument("--report-every", type=int, default=25, help="Print status every N ticks (default: 25)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Generate plots after run (needs metrics)")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory for plots")

    args = parser.parse_args()

    try:
        if args.load:
            state = SimulationState.load_from_file(args.load)

            # Optional overrides on resume
            if args.betrayal != 200:
                state.betrayal_step = args.betrayal
            state.collision_policy = args.collision
            state.shuffle_order = not args.no_shuffle
            state.metrics_every = args.metrics_every
            state.persist_metrics = not args.no_save_metrics

            state = run_simulation(
                state,
                total_steps=args.steps,
                report_every=0 if args.quiet else args.report_every,
                save_path=args.save,
                save_every=args.save_every,
                quiet=args.quiet,
            )
        else:
            state = make_initial_state(
                num_agents=args.agents,
                include_stay=not args.no_stay,
                max_energy=args.max_energy,
                width=args.width,
                height=args.height,
                betrayal_step=args.betrayal,
                collision_policy=args.collision,
                shuffle_order=not args.no_shuffle,
                metrics_every=args.metrics_every,
                persist_metrics=not args.no_save_metrics,
                seed=args.seed,
                food=args.food,
                hazard=args.hazard,
                safe=args.safe,
                danger=args.danger,
                home_center=not args.no_home_center,
            )
            state = run_simulation(
                state,
                total_steps=args.steps,
                report_every=0 if args.quiet else args.report_every,
                save_path=args.save,
                save_every=args.save_every,
                quiet=args.quiet,
            )

        if args.visualize:
            if state.metrics_every == 0:
                print("Visualization requested, but metrics are disabled (metrics_every=0).")
            elif not state.metrics or all(len(t.snapshots) == 0 for t in state.metrics.values()):
                print("Visualization requested, but no metrics data exists. Run without --metrics-every 0.")
            else:
                visualize_simulation(state, output_dir=args.plot_dir)

        if not args.quiet:
            print("\n=== Summary ===")
            for aid, agent in state.agents.items():
                pos = state.env.agent_positions.get(aid)
                tracker = state.metrics.get(aid)
                steps_recorded = len(tracker.snapshots) if tracker else 0
                print(
                    f"[{aid}] "
                    f"alive={not agent.vitality.is_dead} "
                    f"E={agent.vitality.energy:.1f} I={agent.vitality.integrity:.1f} "
                    f"pos=({pos.x if pos else '?'} , {pos.y if pos else '?'}) "
                    f"metrics_steps={steps_recorded}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
