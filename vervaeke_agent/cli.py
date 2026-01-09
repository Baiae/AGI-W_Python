"""Command-line interface for the Vervaeke Agent simulation."""

from __future__ import annotations

import argparse
import sys

from . import sim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vervaeke Agent Simulation v1.3.1 (persistent, multi-agent, metrics, visualization)"
    )

    # Control
    parser.add_argument("--steps", type=int, default=500, help="Steps to run this session")
    parser.add_argument("--betrayal", type=int, default=200, help="Tick where betrayal starts")
    parser.add_argument("--agents", type=int, default=1, help="Number of agents (new run only)")
    parser.add_argument("--width", type=int, default=10, help="Grid width (new run only)")
    parser.add_argument("--height", type=int, default=10, help="Grid height (new run only)")
    parser.add_argument("--max-energy", type=float, default=100.0, help="Energy cap")
    parser.add_argument("--no-stay", action="store_true", help="Disable STAY action")

    # World content knobs (new run only)
    parser.add_argument("--food", type=int, default=8)
    parser.add_argument("--hazard", type=int, default=4)
    parser.add_argument("--safe", type=int, default=6)
    parser.add_argument("--danger", type=int, default=3)
    parser.add_argument(
        "--no-home-center", action="store_true", help="Do not place HOME_MARKER at center"
    )

    # Dynamics knobs
    parser.add_argument("--collision", choices=["allow", "block"], default="allow")
    parser.add_argument(
        "--no-shuffle", action="store_true", help="Disable per-tick agent-order shuffling"
    )
    parser.add_argument(
        "--metrics-every", type=int, default=1, help="Record metrics every N ticks (0 disables)"
    )

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
    parser.add_argument("--report-every", type=int, default=25, help="Print status every N ticks")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    # Visualization
    parser.add_argument(
        "--visualize", action="store_true", help="Generate plots after run (needs metrics)"
    )
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory for plots")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.load:
            state = sim.SimulationState.load_from_file(args.load)

            # Optional overrides on resume
            if args.betrayal != 200:
                state.betrayal_step = args.betrayal
            state.collision_policy = args.collision
            state.shuffle_order = not args.no_shuffle
            state.metrics_every = args.metrics_every
            state.persist_metrics = not args.no_save_metrics

            state = sim.run_simulation(
                state,
                total_steps=args.steps,
                report_every=0 if args.quiet else args.report_every,
                save_path=args.save,
                save_every=args.save_every,
                quiet=args.quiet,
            )
        else:
            state = sim.make_initial_state(
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
            state = sim.run_simulation(
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
                print(
                    "Visualization requested, but no metrics data exists. Run without --metrics-every 0."
                )
            else:
                sim.visualize_simulation(state, output_dir=args.plot_dir)

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
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
