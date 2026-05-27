# DVC Model Artifact Workflow

## Use when

Use this when large datasets, checkpoints, or derived artifacts should be reproducible without committing binaries to Git.

## Inputs

- Artifact paths and expected sizes.
- Remote storage policy.
- Metrics and params files.

## Recipe

1. Track large artifacts with DVC and small configs with Git.
2. Define stages for preprocessing, training, and evaluation when outputs are reproducible.
3. Commit `.dvc`, `dvc.yaml`, `dvc.lock`, params, and metrics metadata.
4. Push artifacts to the configured remote.
5. Document how to recover the exact artifact version.

## Checks

- Do not commit raw checkpoints or datasets accidentally.
- Verify `dvc status` before reporting reproducibility.
- Record model and dataset revisions separately.

## Expected output

A reproducible artifact workflow with tracked metadata, remote path, and recovery command.

## References

- https://dvc.org/doc/user-guide/what-is-dvc
- https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial
