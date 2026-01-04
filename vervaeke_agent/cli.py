"""Command-line interface for the Vervaeke Agent simulation."""

import argparse

from . import sim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vervaeke Agent simulation")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sim.run(steps=args.steps)
