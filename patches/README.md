# Patches

Tracking fixes and patches applied to the codebase for future reference.

---

## 001 — Candy Crush: inject dynamic action mapping into GymEnvAdapter (2026-03-11)

**File:** `GamingAgent/gamingagent/envs/custom_03_candy_crush/candyCrushEnv.py` (line ~266)

**Problem:**
`game_env_config.json` ships with `"action_mapping": {}` because Candy Crush
has 112 valid swap actions on an 8x8 board — too many to enumerate manually
in the config (unlike 2048/Sokoban which have only 4 directional actions).

Because the adapter's `move_to_action_idx` was always empty, every call to
`_parse_agent_action_str` triggered:

1. `adapter.map_agent_action_to_env_action()` → miss → **warning printed**
2. Fallback regex parse of `((r1,c1),(r2,c2))` → lookup in
   `env_move_to_action_idx` → success

Actions executed correctly, but the per-step warning was noisy and the
unnecessary fallback added overhead.

**Fix:**
After `CandyCrushEnv.__init__` dynamically builds `env_move_to_action_idx`
from the board geometry, inject that mapping into the adapter when the
adapter's own mapping (from config) is empty:

```python
if not self.adapter.move_to_action_idx and self.env_move_to_action_idx:
    self.adapter.move_to_action_idx = {k.lower(): v for k, v in self.env_move_to_action_idx.items()}
    self.adapter.action_idx_to_move = {v: k for k, v in self.env_move_to_action_idx.items()}
```

Now the adapter's first lookup succeeds directly — no warning, no regex
fallback needed.
