# Plasmid Priority

Retrospective genomic surveillance prioritization for operational plasmid backbone classes.

## Overview

Plasmid Priority is a bioinformatics framework that scores plasmid backbones
across three orthogonal axes:

- **T** – Transfer efficiency
- **H** – Host range breadth
- **A** – AMR / clinical hazard

These are fused into a single priority index used for retrospective surveillance.

## Quick Start

```bash
pip install -e ".[analysis,dev]"
make quality
```

## Architecture

See the [API Reference](api/exceptions.md) for detailed module documentation.

## Operations

- [Deployment and rollback runbook](runbooks/deployment.md)
- [Release verification runbook](runbooks/release_verification.md)
- [Quality scoreboard](runbooks/quality_scoreboard.md)
- [Scientific invariants](runbooks/scientific_invariants.md)
- [Security runbook](runbooks/security.md)
- [Demo path](runbooks/demo_path.md)

## Stakeholder Overviews

- [Scientific overview](scientific_overview.md)
- [CTO overview](cto_overview.md)
- [Investor overview](investor_overview.md)
- [Product strategy](product_strategy.md)
