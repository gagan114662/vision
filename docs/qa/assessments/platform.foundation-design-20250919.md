# QA Test Design — Platform Foundation
- **Epic**: platform
- **Story**: foundation
- **Date**: 2025-09-19

## Test Objectives
1. Verify documentation completeness and consistency across PRD, architecture, and standards.
2. Ensure BMad `.bmad-core` configuration references existing files and loads required context.
3. Confirm roadmap aligns with staged validation checkpoints and regulatory requirements.

## Proposed Tests
- **T1**: Static analysis script validating presence and non-empty status of required docs (`docs/prd.md`, `docs/architecture.md`, `docs/architecture/*`, `.bmad-core/core-config.yaml`, `.bmad-core/data/technical-preferences.md`).
- **T2**: Lint YAML/Markdown formatting; enforce schema for `.bmad-core/core-config.yaml` and `semtools-metadata.yaml` using yamllint/custom schema.
- **T3**: Check roadmap milestones against PRD goals; automated parser ensures each goal has at least one milestone reference.
- **T4**: Verify QA assessment/gate files exist for foundation story.

## Tooling
- Python pytest suite under `tests/qa/test_docs.py` (TBD).
- Pre-commit hook to run linting and doc validation.

## Acceptance Metrics
- All tests pass in CI; documentation coverage report generated.

## Dependencies
- Story backlog for implementing validation scripts.
- Access to repo CI configuration.
