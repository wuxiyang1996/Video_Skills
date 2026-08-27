"""GRPO collect/train quality filters and diversity metrics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trainer.grpo.advantages import assign_group_advantages
from trainer.grpo.types import GrpoGroup, GrpoRollout
from trainer.reward.verified_reward import VerifiedRewardBreakdown


def answer_label(rollout: GrpoRollout | Mapping[str, Any]) -> str | None:
    if isinstance(rollout, GrpoRollout):
        extras = rollout.extras or {}
        pv = rollout.policy_view or {}
        reward = rollout.reward
    else:
        extras = rollout.get("extras") or {}
        pv = rollout.get("policy_view") or {}
        reward = rollout.get("reward") or {}
    fa = extras.get("final_answer") if isinstance(extras, dict) else None
    if fa is None and isinstance(pv, dict):
        fa = pv.get("final_answer")
    if isinstance(fa, dict):
        label = fa.get("label")
        return str(label).strip() if label not in (None, "") else None
    if fa not in (None, ""):
        return str(fa).strip()
    _ = reward  # keep signature symmetric for future reward-derived labels
    return None


def progress_total(rollout: GrpoRollout | Mapping[str, Any]) -> int:
    if isinstance(rollout, GrpoRollout):
        return int(rollout.reward.progress_total or 0)
    reward = rollout.get("reward") or {}
    return int(reward.get("progress_total") or 0)


def is_dirty_rollout(rollout: GrpoRollout | Mapping[str, Any]) -> bool:
    """Drop empty commits / zero-progress traces from training groups."""
    return answer_label(rollout) is None or progress_total(rollout) <= 0


def filter_group_rollouts(
    group: GrpoGroup,
    *,
    drop_dirty: bool = True,
    min_k: int = 2,
) -> GrpoGroup | None:
    """Keep clean rollouts and recompute advantages; drop group if < min_k remain."""
    kept = [r for r in group.rollouts if not (drop_dirty and is_dirty_rollout(r))]
    if len(kept) < int(min_k):
        return None
    advantages = assign_group_advantages([r.reward for r in kept])
    for rollout, adv in zip(kept, advantages):
        rollout.advantage = float(adv)
    return GrpoGroup(
        group_id=group.group_id,
        example_id=group.example_id,
        video_key=group.video_key,
        split_role=group.split_role,
        mode=group.mode,
        rollouts=kept,
    )


def filter_groups_for_training(
    groups: Sequence[GrpoGroup],
    *,
    drop_dirty: bool = True,
    min_k: int = 2,
) -> list[GrpoGroup]:
    out: list[GrpoGroup] = []
    for group in groups:
        filtered = filter_group_rollouts(group, drop_dirty=drop_dirty, min_k=min_k)
        if filtered is not None:
            out.append(filtered)
    return out


def _reward_from_mapping(payload: Mapping[str, Any]) -> VerifiedRewardBreakdown:
    progress = payload.get("verified_atomic_progress") or (0, 0, 0, 0, 0)
    progress_t = tuple(int(x) for x in progress)
    rank_key = payload.get("rank_key")
    if rank_key is None:
        rank_key = (
            int(payload.get("hard_feasible", 0)),
            int(payload.get("terminal_success", 0)),
            progress_t,
            int(payload.get("evidence_checks", 0)),
            -int(payload.get("cost_total", 0)),
        )
    else:
        rk = list(rank_key)
        if len(rk) >= 3 and isinstance(rk[2], list):
            rk[2] = tuple(int(x) for x in rk[2])
        rank_key = tuple(rk)
    return VerifiedRewardBreakdown(
        spec_version=str(payload.get("spec_version") or "video-skills/verified-reward-v2"),
        hard_feasible=bool(payload.get("hard_feasible")),
        terminal_success=bool(payload.get("terminal_success")),
        verified_atomic_progress=progress_t,
        progress_total=int(payload.get("progress_total") or sum(progress_t)),
        evidence_checks=int(payload.get("evidence_checks") or 0),
        cost_total=int(payload.get("cost_total") or 0),
        rank_key=rank_key,  # type: ignore[arg-type]
        hard_failures=tuple(payload.get("hard_failures") or ()),
        blocked_strong_commit=bool(payload.get("blocked_strong_commit")),
    )


def filter_group_dicts_for_training(
    groups: Sequence[Mapping[str, Any]],
    *,
    drop_dirty: bool = True,
    min_k: int = 2,
) -> list[dict[str, Any]]:
    """Train-path filter for JSONL-loaded groups (recomputes rank advantages)."""
    from trainer.reward import group_rank_advantages

    cleaned: list[dict[str, Any]] = []
    for group in groups:
        rollouts = list(group.get("rollouts") or [])
        kept = [r for r in rollouts if not (drop_dirty and is_dirty_rollout(r))]
        if len(kept) < int(min_k):
            continue
        rewards = [_reward_from_mapping(r.get("reward") or {}) for r in kept]
        advantages = group_rank_advantages(rewards)
        new_rollouts = []
        for r, adv in zip(kept, advantages):
            row = dict(r)
            row["advantage"] = float(adv)
            new_rollouts.append(row)
        cleaned.append({**dict(group), "rollouts": new_rollouts})
    return cleaned


def summarize_group_quality(groups: Sequence[GrpoGroup | Mapping[str, Any]]) -> dict[str, Any]:
    """Diversity / dirtiness metrics for collect_summary + gates."""
    n_groups = 0
    label_div = 0
    motif_div = 0
    collapsed = 0
    empty_answer = 0
    dirty_samples = 0
    total_samples = 0
    terminal = 0
    nontrivial_adv_groups = 0
    compare_ran_samples = 0

    for group in groups:
        n_groups += 1
        if isinstance(group, GrpoGroup):
            rollouts: Sequence[Any] = group.rollouts
            example_id = group.example_id
        else:
            rollouts = list(group.get("rollouts") or [])
            example_id = str(group.get("example_id") or "")

        labels: list[str | None] = []
        motifs: list[str | None] = []
        advs: list[float] = []
        for r in rollouts:
            total_samples += 1
            labels.append(answer_label(r))
            if is_dirty_rollout(r):
                dirty_samples += 1
            if isinstance(r, GrpoRollout):
                mo = r.motif_online or {}
                advs.append(float(r.advantage))
                terminal += int(bool(r.reward.terminal_success))
                pv = r.policy_view or {}
            else:
                mo = r.get("motif_online") or {}
                advs.append(float(r.get("advantage") or 0.0))
                terminal += int(bool((r.get("reward") or {}).get("terminal_success")))
                pv = r.get("policy_view") or {}
            motifs.append(mo.get("selected_motif_id") or mo.get("motif_id"))
            meta = pv.get("metadata") if isinstance(pv, dict) else {}
            meta = meta if isinstance(meta, dict) else {}
            skills = meta.get("executed_skill_ids") or []
            if meta.get("compare_hypotheses_ran") or "compare_hypotheses" in skills:
                compare_ran_samples += 1

        non_null = {x for x in labels if x not in (None, "")}
        if len(non_null) > 1:
            label_div += 1
        elif len(non_null) == 1 and all(x not in (None, "") for x in labels):
            collapsed += 1
        if all(x in (None, "") for x in labels):
            empty_answer += 1
        if len({m for m in motifs if m}) > 1:
            motif_div += 1
        if advs and (max(advs) - min(advs)) > 1e-9:
            nontrivial_adv_groups += 1
        _ = example_id

    compare_rate = compare_ran_samples / max(total_samples, 1)
    return {
        "n_groups": n_groups,
        "n_samples": total_samples,
        "label_diverse_groups": label_div,
        "label_diverse_rate": label_div / max(n_groups, 1),
        "label_collapsed_groups": collapsed,
        "label_collapsed_rate": collapsed / max(n_groups, 1),
        "motif_diverse_groups": motif_div,
        "motif_diverse_rate": motif_div / max(n_groups, 1),
        "empty_answer_groups": empty_answer,
        "empty_answer_rate": empty_answer / max(n_groups, 1),
        "dirty_samples": dirty_samples,
        "dirty_sample_rate": dirty_samples / max(total_samples, 1),
        "mean_terminal_success": terminal / max(total_samples, 1),
        "nontrivial_advantage_groups": nontrivial_adv_groups,
        "nontrivial_advantage_rate": nontrivial_adv_groups / max(n_groups, 1),
        "compare_hypotheses_coverage": compare_rate,
        "gates": {
            "label_diverse_ge_0_60": (label_div / max(n_groups, 1)) >= 0.60,
            "mean_terminal_ge_0_15": (terminal / max(total_samples, 1)) >= 0.15,
            "empty_answer_lt_0_10": (empty_answer / max(n_groups, 1)) < 0.10,
            "nontrivial_adv_ge_0_50": (nontrivial_adv_groups / max(n_groups, 1)) >= 0.50,
            "compare_coverage_ge_0_90": compare_rate >= 0.90,
        },
    }
