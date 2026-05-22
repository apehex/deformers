# Agent Skills

This directory is reserved for reusable skills that help agents perform recurring work in this repository.

Add a skill when a workflow needs more durable guidance than a one-off note in `docs/`, such as a repeated experiment pattern, evaluation protocol, model-analysis routine, or documentation workflow.

## Layout

Each skill should live in its own directory:

```text
.agents/skills/
  example-skill/
    SKILL.md
    references/
    scripts/
```

Use `TEMPLATE.md` as the starting point for each `SKILL.md`.

## Skill Rules

- Keep skills narrow and action-oriented.
- State when the skill should and should not be used.
- Prefer explicit inputs, outputs, and verification steps.
- Link to stable repository docs instead of duplicating long context.
- Include scripts or references only when they materially reduce repeated work.
- Keep examples safe, reproducible, and free of operationally harmful content.
