# Deployment and Rollback Runbook

This runbook covers the standard path for deploying a new pipeline revision and
the fastest rollback path if a release regresses.

## Preconditions

- A clean working tree or a release branch with the intended changes.
- The data contract and config layers validate locally.
- The latest pipeline checkpoint directory is intact if you plan to resume a prior run.

## Deployment

1. Run the targeted test suite before publishing.
2. Run the canonical release verification gauntlet and keep the generated summary.
3. Build the branch outputs with the workflow runner.
4. Confirm the checkpoint files were written and the script summaries contain
   non-empty `output_manifest` entries.
5. Verify the generated reports and manifests in `data/*/analysis/`.
6. Publish only after the smoke test and validation scripts succeed.

Recommended commands:

```bash
uv run pytest -q tests/test_config.py tests/test_properties.py tests/test_workflow.py -o addopts=''
make verify-release
uv run python scripts/run_workflow.py release --max-workers 4
```

## Rollback

Use the rollback path when a new run produces a bad manifest, an invalid schema,
or a materially different report.

1. Stop any active pipeline execution.
2. Restore the last known-good artifacts from the prior manifest or release bundle.
3. Re-run the affected step with `--no-resume` if the checkpoint state is suspect.
4. Re-run the validation and smoke scripts against the restored outputs.
5. Only re-enable downstream publishing after the restored outputs match the
   prior manifest and the checks pass.

Rollback confirmation command:

```bash
make verify-release
```

## Common Failure Modes

- Corrupted checkpoint JSON: delete the checkpoint file and re-run with
  `--no-resume`.
- Stale cached outputs: remove the affected analysis directory and rebuild the
  branch.
- Schema validation failure: fix the upstream artifact and regenerate every
  downstream output that depends on it.
- Unexpected timeout: inspect the step summary, then rerun with a larger
  `PLASMID_PRIORITY_STEP_TIMEOUT_SECONDS` value if the step is still valid.

## Ownership

- Pipeline integrity and rollback decisions should be owned by the release
  operator on call.
- Data and schema validation failures should be triaged by the branch owner.
