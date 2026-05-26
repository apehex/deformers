---
name: literature-and-evaluation-tooling
description: Use for literature retrieval, citation metadata, benchmark/eval harness selection, LLM evaluation, red-team tool selection, evidence collection, redaction, and responsible reporting workflows.
---

# Literature and Evaluation Tooling

## When to Use

- Use this skill when searching papers, collecting citation metadata, building a literature table, selecting benchmarks, running evaluation harnesses, or preparing evidence reports.
- Use it for arXiv, Semantic Scholar, CrossRef, PubMed/Europe PMC/CORE, BibTeX/Zotero-style reference handling, LM-Eval-Harness, OpenAI Evals, PromptFoo, DeepTeam, Garak, PyRIT, and benchmark CI.
- Do not use it to generate live harmful prompts or operational exploit details; use safe placeholders for unsafe targets.

## Inputs

- Research topic, hypothesis, or benchmark target.
- Model/system under evaluation.
- Desired evidence: paper list, benchmark scores, red-team report, citation map, or regression suite.
- Scope and authorization for any safety or red-team work.

## Tool Selection

- Use arXiv for preprints and broad topic search.
- Use Semantic Scholar for citation graphs, related papers, author metadata, and filtering by venue/year/citations.
- Use CrossRef for DOI and publication metadata.
- Use PubMed/Europe PMC/CORE for biomedical or domain-specific literature.
- Use LM-Eval-Harness for standard open LLM benchmarks and local model evaluation.
- Use OpenAI Evals or repo-native pytest-style evals for custom task definitions and regression tests.
- Use PromptFoo, DeepTeam, Garak, or PyRIT only within authorized defensive scopes and with sanitized artifacts.

## Workflow

1. Define the information need or evaluation question.
2. Search primary sources first; collect title, year, venue, URL/DOI, model/task, and key claim.
3. Deduplicate by DOI, arXiv ID, or normalized title.
4. For benchmarks, pin model revision, tokenizer/template, task version, seed, decoding settings, and metric definitions.
5. For red-team evaluations, create an authorization packet and safe run sheet before execution.
6. Redact sensitive or operationally harmful outputs before summaries, issues, or reports.
7. Record sources in `docs/references.md` and results in `logs/` when the work is non-trivial.

## Design Checks

- Do not rely on uncited summaries when the exact claim matters.
- Separate benchmark failures from harness bugs, prompt-template bugs, and metric bugs.
- Include negative and benign controls in safety evaluations.
- Preserve raw evidence only in approved locations; reports should use safe placeholders such as `POLICY_BOUNDARY_CROSSING`.
- Avoid novelty claims until related work has been checked.

## Outputs

- Literature table or annotated bibliography.
- Evaluation plan or benchmark command set.
- Sanitized evidence summary with scope, model version, metrics, failures, and limitations.
- Disclosure-ready report when applicable.

## Verification

- Confirm every cited claim has a primary source link or local reference.
- Run a tiny benchmark/eval subset before scaling.
- Check redaction before sharing logs or reports.

## Related Skills

- `universal-jailbreak-research-protocol` for authorized bounty-facing evaluation and disclosure.
- `universal-attack-hypothesis-generation` for mechanism-level hypotheses in authorized jailbreak research.
- `authority-and-role-mechanisms`, `refusal-and-policy-geometry`, and `template-anchoring-and-hidden-policy-carriers` for mechanism-specific safety analysis.

## Common Tool Recipes

- Literature retrieval: search arXiv for preprints, Semantic Scholar for citation graphs, CrossRef for DOI metadata, and PubMed/Europe PMC/CORE for domain-specific corpora.
- Reference handling: normalize by DOI/arXiv ID/title, export BibTeX when useful, and keep source URLs with every claim.
- Benchmarking: use LM-Eval-Harness for standard open LLM benchmarks; use OpenAI Evals or repo-native pytest-style evals for custom tasks and regressions.
- Red-team tooling: use PromptFoo, DeepTeam, Garak, or PyRIT only with authorization, scope, safe placeholders, redaction rules, and sanitized summaries.
- Evidence collection: log prompt/template/model revision/decoding settings/metric version; preserve raw evidence only in approved locations.
- Reporting: separate benchmark failures, harness bugs, prompt-template bugs, and metric bugs; include limitations and reproduction steps.
