# Motif

This package is the video-native motif layer for L1/L2 graph agents.

A motif is a reusable L1/L2 graph template, not an executable skill. L1/L2
agents may retrieve a motif as a planning prior, but every motif must expand
back into ordinary L1 evidence nodes and L2 reasoning nodes before verifier
checks and answer emission.

## Definition

A motif is a reusable, verified L1/L2 subgraph prior.

It is not a segment cut from a raw trajectory. It is a canonical graph pattern
mined from accepted or verifier-supported L1/L2 rollouts, with enough evidence
to be reused as a planning or repair prior.

A valid motif should contain:

- an L2 reasoning subgraph, and usually its referenced L1 evidence
  neighborhood;
- abstract evidence roles, not copied video facts;
- argument-binding templates for entities, timestamps, answer options, claims,
  and evidence refs;
- an expansion template that maps back to frozen L1/L2 atomic nodes;
- support, failure, transfer, and verifier statistics.

A motif is not:

- a new atomic skill id;
- a callable skill agent;
- a black-box executor;
- a benchmark-specific answer shortcut;
- persistent evidence from older videos;
- a way to bypass node-level verification.

Every runtime use must expand into ordinary L1 evidence nodes and L2 reasoning
nodes before execution, verification, repair, and answer emission.

## Intended Flow

1. Mine candidate motifs from successful or partially successful L1/L2 rollouts.
2. Use Qwen3.5 / GPT-OSS as proposer and curator agents.
3. Store candidates in `MotifBank` with supporting L1/L2 evidence refs.
4. Keep new motifs in `candidate` or `shadow` until transfer checks pass.
5. Run heldout same-domain transfer evaluation:
   - baseline: L1/L2 only
   - treatment: L1/L2 plus motif prior
6. Promote only motifs that preserve hard evaluation correctness, evidence
   validity, verifier support, and no-leakage constraints.

## Boundary

Evaluation remains hard 0/1. Progressive motif reward is only for training or
curation:

- final answer correctness
- evidence validity
- verifier support
- no hidden-supervision leakage
- non-regression against L1/L2-only baseline

## COS-PLAY Mapping

This module borrows the useful structure from COS-PLAY skill agents:

- skill bank -> motif bank
- skill lifecycle -> motif lifecycle
- query skill -> query motif
- skill-following reward -> motif-following progressive RLVR
- cross-game archetype -> cross-rollout L1/L2 motif

It does not copy game trajectory segmentation or executable skill runtime.

## Motif Extraction Is Not Trajectory Segmentation

Older skill-agent pipelines depend on heuristic trajectory segmentation signals
such as phase changes, boundary scores, predicate transitions, or temporal
windows. The L1/L2 motif layer should not copy that assumption.

The correct unit is a verified graph fragment:

```text
accepted L1/L2 rollout
  -> choose anchor node: compare / verify / commit / infer_* / repair
  -> take dependency closure over L2 nodes
  -> take evidence closure over referenced L1 nodes
  -> canonicalize entities, timestamps, options, and local facts into roles
  -> check that the fragment expands into frozen L1/L2 atomic nodes
  -> store as candidate motif
  -> promote only after support, transfer, verifier, and no-leakage gates
```

This means the miner should prefer graph-native cuts:

- dependency closure around a reasoning anchor;
- evidence closure from `evidence_refs`;
- claim closure from supported claims to verifier and commit nodes;
- branch closure for repeated hypothesis retrieval/scoring branches;
- repair closure around failed nodes and localized fixes.

The first deterministic miner can surface rough candidates, but only candidates
that survive canonicalization, expansion validation, and transfer gates should
be called real motifs.

## Existing L1/L2 Mining Smoke Test

Run deterministic mining over saved L1/L2 outputs:

```bash
python -m dataset_clip_wrapper.motifs.mine_existing_l1_l2 \
  /fs/gamma-projects/vlm-robot/video_skills_relaunched/dataset_clip_wrapper/output/batch3_cg_strict_qwen.jsonl \
  /fs/gamma-projects/vlm-robot/video_skills_relaunched/dataset_clip_wrapper/output/batch3_video_holmes_strict_qwen.jsonl \
  /fs/gamma-projects/vlm-robot/video_skills_relaunched/dataset_clip_wrapper/output/batch3_videomme_strict_qwen.jsonl \
  /fs/gamma-projects/vlm-robot/video_skills_relaunched/dataset_clip_wrapper/output/batch3_vr_strict_qwen.jsonl \
  --output-bank dataset_clip_wrapper/motifs/output/batch3_strict_qwen_motif_bank.jsonl \
  --summary-output dataset_clip_wrapper/motifs/output/batch3_strict_qwen_motif_summary.json \
  --min-support 2
```

The first smoke run found reusable structure, but it should be treated as
candidate evidence rather than a final motif bank:

- `Claim Support To Commit`: repeated across short and long Video QA rollouts.
- `Hypothesis Fanout And Compare`: repeated in multiple-choice whole-video /
  streaming runs.
- Long-video coarse-to-fine L2 sequence: repeated across CG-Bench / VRBench
  style long-video cases.
- L1 evidence profiles: useful as audit signals, but too statistical to be
  promoted without Qwen3.5 / GPT-OSS curator rewriting them into semantic
  graph templates.

Interpretation:

- `Claim Support To Commit` is close to a motif, but still needs a role schema
  for claim, support refs, verifier node, and commit node.
- `Hypothesis Fanout And Compare` is close to a motif for multiple-choice QA,
  especially when branch roles can be canonicalized.
- Long full L2 sequences are usually too coarse; they should be split into
  smaller connected subgraph motifs.
- L1 evidence profiles are not motifs by themselves. They are mining/audit
  signals that may help a curator propose semantic evidence-role templates.

Generated smoke outputs are written under `dataset_clip_wrapper/motifs/output/`.
