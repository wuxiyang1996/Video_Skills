# Skill Bank — Persistent Storage for Skill Contracts

**Location:** `skill_agents/skill_bank/`

## Overview

The Skill Bank provides **persistent storage** for learned skill contracts. The main class is **SkillBankMVP**, which stores effects-only contracts from [Stage 3 MVP](../stage3_mvp/README.md):

- `SkillEffectsContract` per skill (eff_add, eff_del, eff_event)
- Optional `VerificationReport` per skill
- Versioning and JSONL persistence
- Mutation history for reproducibility

Used by the full [skill_agents](../README.md) pipeline, [contract_verification](../contract_verification/README.md), [bank_maintenance](../bank_maintenance/README.md), and [skill_evaluation](../skill_evaluation/README.md).

## High-level API

```python
from skill_agents.skill_bank import SkillBankMVP

bank = SkillBankMVP(path="data/skill_bank.jsonl")
bank.load()

# Queries
bank.skill_ids
bank.get_contract("nav_to_pot")
bank.get_report("nav_to_pot")
bank.has_skill("nav_to_pot")

# Mutations (usually via Stage 3 or bank_maintenance)
bank.add_or_update(contract, report=...)
bank.remove("old_skill")

bank.save()
```

## Key module

| Module | Purpose |
|--------|--------|
| `bank` | `SkillBankMVP`: in-memory contracts + reports, JSONL load/save, version log |
