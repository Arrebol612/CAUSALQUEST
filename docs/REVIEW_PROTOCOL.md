# Manual review protocol

Sampling writes an immutable `run.json`, a public `review_packet.json`, and `review_template.json`.
Copy the template outside the run directory. Keep the run, packet, rubric and episode hashes unchanged.
Review every fixed episode, including failed submissions. An edited template never executes a model.

For each submitted report:

- Set `status` to `scored`, identify `reviewer_id`, and explain the judgment in `note`.
- Fill each task-applicable dimension with integer 0, 1 or 2, and provide its `reasons` entry.
- Keep inapplicable scores null. A suspected but unresolved interpretation remains pending.
- Set `core_supported` using object, decisive relation, main recommendation and actual next action.
- Record material hard-boundary violations in `hard_violations`. Do not promote a dangerous answer
  because it has a correct number or a valid decision enum.
- Cite `evidence_ids` from actual successful observations. Programmatic existence checks are not entailment checks.

Tiers: hard violation -> 0; unsupported core -> 1; supported core with a material remaining defect -> 2;
all applicable dimensions fully supported -> 3. A terminal non-submission has the runtime's fixed tier-0
record with unassessed semantic dimensions null. Leave that record unchanged.

```bash
cq reviews check --run outputs/round1 --input local_reviews/round1.json
cq reviews import --run outputs/round1 --input local_reviews/round1.json
```

Missing, duplicated, pending, contradictory, mismatched or unsupported-reference entries block import.
No partially reviewed batch trains. The full population remains the metric denominator. A program cannot
verify the truth of a human's explanation; reviewer calibration, adjudication and blinded comparison
are separate research obligations.

The toy-only scorer exists to test workflows. `cq reviews toy-score` rejects real CausalQuest task kinds.
It must not be used as an experiment decision reward model. Deterministic arithmetic and structural
checks can assist a reviewer, but the current CLI does not certify all seven semantic dimensions.

