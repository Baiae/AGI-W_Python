# Vervaeke Agent

Headless simulation scaffold for the Vervaeke Agent project.

## Quickstart

```bash
bash scripts/bootstrap.sh
```

After bootstrap, run commands using the venv interpreter to ensure dependencies
are available:

```bash
.venv/bin/python -m vervaeke_agent --steps 100
bash scripts/run.sh --steps 200 --agents 3
```

> ⚠️ Running `python -m vervaeke_agent` outside the venv may fail if global
> dependencies (like `pydantic`) are not installed.

## Checks

```bash
bash scripts/check.sh
```
