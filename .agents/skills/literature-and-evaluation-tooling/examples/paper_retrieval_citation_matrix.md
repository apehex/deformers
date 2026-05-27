# Paper Retrieval and Citation Matrix

## Use when

Use this when turning a research question into a sourced reading list and evidence table.

## Inputs

- Research question and inclusion criteria.
- Seed papers or local references.
- Preferred citation format.

## Recipe

1. Search primary sources first: arXiv, OpenReview, ACL Anthology, official docs, or project repos.
2. Record title, authors, date, URL, method, dataset, and claim.
3. Split evidence into direct support, related background, contradiction, and tool reference.
4. Mark unreviewed blog posts or secondary summaries as lower confidence.
5. Update local references only when the paper materially affects a repo decision.

## Checks

- Do not claim novelty before checking close prior work.
- Verify equations against PDFs when converted markdown looks corrupted.
- Include negative or contradicting papers.

## Expected output

A citation matrix that lets another agent justify source selection and identify gaps.

## References

- https://openreview.net/
- https://arxiv.org/
- https://aclanthology.org/
