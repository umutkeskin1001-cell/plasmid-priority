# Security Runbook

This is a read-only analytical API and release surface. Security work here is about reducing accidental exposure and release drift.

## Current Controls

- optional API key guard on the scoring surface
- request size limits
- per-route rate limiting
- dependency audit via `make security`
- strict generated-artifact validation before release claims

The API protection layer is implemented in `src/plasmid_priority/api/app.py`.

## Operator Commands

```bash
make security
make verify-release
```

## Release Expectations

- no write endpoints
- no anonymous release publishing path
- no release claim without passing dependency and artifact gates

## Incident Handling

1. Revoke the API key if one is configured.
2. Stop serving the API surface.
3. Run `make verify-release` after the fix.
4. Rebuild release artifacts only after the verification gauntlet passes.
