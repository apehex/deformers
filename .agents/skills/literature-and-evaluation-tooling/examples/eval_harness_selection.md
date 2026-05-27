# Evaluation Harness Selection

## Use when

Use this when choosing between reusable LLM evaluation frameworks.

## Inputs

- Model access mode: local HF, vLLM, API, agent, or custom pipeline.
- Task type: logprob benchmark, generative grading, safety probe, or agent task.
- Reproducibility requirements.

## Recipe

1. Use `lm-evaluation-harness` for standard autoregressive benchmarks and task registry workflows.
2. Use OpenAI Evals or platform evals when building custom model/system evals around OpenAI APIs.
3. Use Inspect AI for safety, agent, and solver/scorer-style evaluation tasks.
4. Use HELM when broad, transparent, multi-metric benchmark reporting matters more than lightweight iteration.
5. Always log model id, date, decoding settings, sample outputs, and scorer version.

## Checks

- Validate task configs before running large jobs.
- Keep custom grading prompts versioned.
- Avoid comparing scores produced by incompatible harness settings.

## Expected output

A harness choice note with command skeleton, task config location, metrics, and reproducibility fields.

## References

- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
- https://github.com/openai/evals
- https://inspect.ai-safety-institute.org.uk/
- https://crfm-helm.readthedocs.io/
