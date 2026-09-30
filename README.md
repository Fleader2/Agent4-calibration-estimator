# Agent 4 — Calibration and Parameter Estimation

## Overview

Agent 4 is the fourth component of a multi-agent system for constructing scientifically
traceable, mechanistic models of microbial metabolism.

Agent 4 consumes Agent 2's model/parameter metadata and Agent 3's simulation/diagnostic report
(two separate repositories, `agent2-antimony-builder` and `agent3-simulation-diagnostics`),
calibrates only explicitly authorized uncertain parameters/initial conditions against supplied
observation data, and produces a provenance-preserving calibrated result for **Agent 5** (not
yet built).

**Central rule:** Agent 4 may estimate uncertain parameters and initial conditions from supplied
observation data, but must never overwrite curated/evidence-backed values or invent
observations.

Agent 4 does **not**:

- curate literature or biological database records (Agent 1's job),
- assemble a model or declare its kinetic laws/parameters (Agent 2's job),
- simulate a model or diagnose numerical pathology from scratch (Agent 3's job — Agent 4 reuses
  the same antimony/libRoadRunner stack Agent 3 already validated),
- perform formal structural identifiability analysis or Bayesian/MCMC estimation in Version 1,
- critique a model's biological plausibility in the full sense Agent 5 will.

Agent 4 consumes Agent 2's and Agent 3's output through an explicit, versioned handoff contract
(a plain, JSON-serializable dict) — it never imports either sibling package's runtime code.

See `docs/01_agent4_responsibility_boundary.md`, `docs/02_agent2_agent3_to_agent4_contract.md`,
`docs/03_calibration_target_eligibility_policy.md`, `docs/04_provenance_semantics.md`, and
`docs/05_identifiability_diagnostics.md` for full detail.

## Canonical workflow

```text
Agent 2 model + Agent 3 diagnostics + observations
        ↓
Choose authorized targets (eligibility policy)
        ↓
Baseline simulation / objective
        ↓
Deterministic, bounded, multi-start optimization
        ↓
Validation simulation
        ↓
Residual / identifiability diagnostics
        ↓
Agent4Report
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Dependencies: `antimony`, `libroadrunner`, `python-libsbml`, `numpy`, `scipy` — the same
simulation stack Agent 3 already validated, plus `scipy.optimize` for deterministic bounded
optimization. No Tellurium, no Bayesian/MCMC dependency.

## Tests

```bash
.venv/bin/python3 -m pytest tests/ -q
```
