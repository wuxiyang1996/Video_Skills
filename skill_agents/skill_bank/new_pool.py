"""
NEW-pool manager: accumulate, cluster, and promote ``__NEW__`` segments.

Instead of treating ``__NEW__`` as a catch-all bucket, the ``NewPoolManager``
tracks rich metadata per candidate, clusters by effect similarity (not exact
signature match), and promotes clusters that meet support, consistency, and
separability criteria.

Usage::

    pool = NewPoolManager(config=NewPoolConfig())
    pool.add(record)                    # accumulate
    candidates = pool.get_candidates()  # inspect mature clusters
    promoted = pool.promote(bank, observations_by_traj)  # materialize

Design principles:
  - Modular: all NEW logic in one place (not scattered across pipeline.py).
  - Reuses existing clustering from ``contract_verification.clustering`` when
    available; falls back to effect-signature Jaccard otherwise.
  - Configurable thresholds for ablation experiments.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from skill_agents.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class NewPoolConfig:
    """Thresholds for NEW-pool management and promotion."""

    # Minimum cluster size to consider for promotion
    min_cluster_size: int = 5

    # Minimum pass rate from Stage 3 verification to promote
    min_pass_rate: float = 0.7

    # Minimum Jaccard distance from any existing skill to promote
    min_distinctiveness: float = 0.25

    # Minimum consistency: fraction of cluster members sharing the majority
    # effect signature pattern
    min_consistency: float = 0.5

    # Jaccard similarity threshold for merging NEW candidates into one cluster
    cluster_similarity_thresh: float = 0.4

    # Maximum number of candidates to hold before forcing a clustering pass
    max_pool_size: int = 500

    # How many top clusters to attempt promotion per call
    max_promotions_per_call: int = 10


# ── Candidate metadata ──────────────────────────────────────────────

@dataclass
class NewCandidate:
    """Rich metadata for one ``__NEW__`` segment candidate."""

    record: SegmentRecord
    effect_sig: str = ""
    duration: int = 0
    predecessor_skill: Optional[str] = None
    successor_skill: Optional[str] = None
    added_at: float = field(default_factory=time.time)

    def __post_init__(self):
        self.effect_sig = self.record.effect_signature()
        self.duration = max(1, self.record.t_end - self.record.t_start + 1)


@dataclass
class ClusterSummary:
    """Summary statistics for one NEW candidate cluster."""

    cluster_id: int
    size: int
    effect_centroid_add: Set[str]
    effect_centroid_del: Set[str]
    effect_centroid_event: Set[str]
    mean_duration: float
    std_duration: float
    consistency: float  # fraction sharing majority effect pattern
    representative_sig: str
    predecessor_distribution: Dict[str, int] = field(default_factory=dict)
    successor_distribution: Dict[str, int] = field(default_factory=dict)


# ── Pool manager ─────────────────────────────────────────────────────

class NewPoolManager:
    """Accumulates, clusters, and promotes ``__NEW__`` segment candidates.

    Replaces the scattered NEW handling in ``pipeline.py`` with a single
    modular component.
    """

    def __init__(self, config: Optional[NewPoolConfig] = None) -> None:
        self.config = config or NewPoolConfig()
        self._candidates: List[NewCandidate] = []
        self._promoted_ids: Set[str] = set()  # seg_ids already promoted
        self._cluster_labels: Optional[np.ndarray] = None
        self._cluster_summaries: List[ClusterSummary] = []

    @property
    def size(self) -> int:
        return len(self._candidates)

    @property
    def records(self) -> List[SegmentRecord]:
        return [c.record for c in self._candidates]

    # ── Accumulate ───────────────────────────────────────────────────

    def add(
        self,
        record: SegmentRecord,
        predecessor_skill: Optional[str] = None,
        successor_skill: Optional[str] = None,
    ) -> None:
        """Add a ``__NEW__`` segment to the pool with context metadata."""
        if record.seg_id in self._promoted_ids:
            return
        cand = NewCandidate(
            record=record,
            predecessor_skill=predecessor_skill,
            successor_skill=successor_skill,
        )
        self._candidates.append(cand)
        self._cluster_labels = None  # invalidate

    def add_batch(
        self,
        records: List[SegmentRecord],
        context: Optional[List[Tuple[Optional[str], Optional[str]]]] = None,
    ) -> None:
        """Add multiple records.  ``context`` is list of (pred_skill, succ_skill)."""
        for i, rec in enumerate(records):
            pred, succ = (None, None) if context is None else context[i]
            self.add(rec, predecessor_skill=pred, successor_skill=succ)

    def remove_promoted(self, seg_ids: Set[str]) -> None:
        """Remove promoted records from the pool."""
        self._promoted_ids |= seg_ids
        self._candidates = [c for c in self._candidates if c.record.seg_id not in seg_ids]
        self._cluster_labels = None

    # ── Clustering ───────────────────────────────────────────────────

    def _build_effect_vectors(self) -> Tuple[np.ndarray, List[str]]:
        """Build binary effect vectors for all candidates."""
        vocab: Set[str] = set()
        for c in self._candidates:
            vocab.update(c.record.eff_add)
            vocab.update(c.record.eff_del)
            vocab.update(c.record.eff_event)
        vocab_list = sorted(vocab)
        if not vocab_list:
            return np.zeros((len(self._candidates), 0), dtype=np.float32), []

        pred_to_idx = {p: i for i, p in enumerate(vocab_list)}
        N = len(self._candidates)
        V = len(vocab_list)
        mat = np.zeros((N, V), dtype=np.float32)
        for row, cand in enumerate(self._candidates):
            for p in cand.record.eff_add:
                if p in pred_to_idx:
                    mat[row, pred_to_idx[p]] = 1.0
            for p in cand.record.eff_del:
                if p in pred_to_idx:
                    mat[row, pred_to_idx[p]] = 1.0
            for p in cand.record.eff_event:
                if p in pred_to_idx:
                    mat[row, pred_to_idx[p]] = 1.0
        return mat, vocab_list

    def cluster(self) -> List[ClusterSummary]:
        """Cluster candidates by effect similarity (Jaccard-based).

        Uses agglomerative clustering from sklearn when available,
        falls back to signature-based grouping otherwise.
        """
        if len(self._candidates) < 2:
            if self._candidates:
                self._cluster_labels = np.zeros(1, dtype=int)
                self._cluster_summaries = [self._build_summary(0, list(range(1)))]
            return self._cluster_summaries

        mat, vocab = self._build_effect_vectors()
        if mat.shape[1] == 0:
            self._cluster_labels = np.zeros(len(self._candidates), dtype=int)
            self._cluster_summaries = [self._build_summary(0, list(range(len(self._candidates))))]
            return self._cluster_summaries

        try:
            labels, n_clusters = self._agglomerative_cluster(mat)
        except ImportError:
            labels, n_clusters = self._signature_cluster()

        self._cluster_labels = labels

        clusters_idx: Dict[int, List[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            clusters_idx[int(lbl)].append(i)

        self._cluster_summaries = []
        for cid in sorted(clusters_idx.keys()):
            self._cluster_summaries.append(
                self._build_summary(cid, clusters_idx[cid])
            )

        return self._cluster_summaries

    def _agglomerative_cluster(self, mat: np.ndarray) -> Tuple[np.ndarray, int]:
        from sklearn.cluster import AgglomerativeClustering

        N = mat.shape[0]
        dist = self._jaccard_distance_matrix(mat)
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - self.config.cluster_similarity_thresh,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist)
        return labels, len(set(labels))

    def _signature_cluster(self) -> Tuple[np.ndarray, int]:
        """Fallback: group by exact effect signature."""
        sig_to_id: Dict[str, int] = {}
        labels = []
        for cand in self._candidates:
            sig = cand.effect_sig
            if sig not in sig_to_id:
                sig_to_id[sig] = len(sig_to_id)
            labels.append(sig_to_id[sig])
        return np.array(labels, dtype=int), len(sig_to_id)

    @staticmethod
    def _jaccard_distance_matrix(mat: np.ndarray) -> np.ndarray:
        N = mat.shape[0]
        dist = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                union = np.sum((mat[i] + mat[j]) > 0)
                inter = np.sum((mat[i] * mat[j]) > 0)
                d = 1.0 - (inter / union) if union > 0 else 0.0
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def _build_summary(self, cid: int, indices: List[int]) -> ClusterSummary:
        """Build summary for one cluster."""
        cands = [self._candidates[i] for i in indices]
        n = len(cands)

        # Effect centroid: literals appearing in >= 50% of instances
        add_counter: Counter = Counter()
        del_counter: Counter = Counter()
        evt_counter: Counter = Counter()
        sig_counter: Counter = Counter()
        pred_dist: Counter = Counter()
        succ_dist: Counter = Counter()
        durations = []

        for c in cands:
            for p in c.record.eff_add:
                add_counter[p] += 1
            for p in c.record.eff_del:
                del_counter[p] += 1
            for p in c.record.eff_event:
                evt_counter[p] += 1
            sig_counter[c.effect_sig] += 1
            if c.predecessor_skill:
                pred_dist[c.predecessor_skill] += 1
            if c.successor_skill:
                succ_dist[c.successor_skill] += 1
            durations.append(c.duration)

        thresh = n * 0.5
        centroid_add = {p for p, c in add_counter.items() if c >= thresh}
        centroid_del = {p for p, c in del_counter.items() if c >= thresh}
        centroid_evt = {p for p, c in evt_counter.items() if c >= thresh}

        most_common_sig, most_common_count = sig_counter.most_common(1)[0]
        consistency = most_common_count / n

        dur_arr = np.array(durations, dtype=np.float64)

        return ClusterSummary(
            cluster_id=cid,
            size=n,
            effect_centroid_add=centroid_add,
            effect_centroid_del=centroid_del,
            effect_centroid_event=centroid_evt,
            mean_duration=float(dur_arr.mean()),
            std_duration=float(dur_arr.std()),
            consistency=consistency,
            representative_sig=most_common_sig,
            predecessor_distribution=dict(pred_dist),
            successor_distribution=dict(succ_dist),
        )

    # ── Candidate inspection ─────────────────────────────────────────

    def get_candidates(self, min_size: Optional[int] = None) -> List[ClusterSummary]:
        """Return cluster summaries for mature candidates.

        Runs clustering if needed.  Filters to clusters with
        ``size >= min_size`` (defaults to ``config.min_cluster_size``).
        """
        if self._cluster_labels is None or len(self._cluster_summaries) == 0:
            self.cluster()

        cutoff = min_size or self.config.min_cluster_size
        return [s for s in self._cluster_summaries if s.size >= cutoff]

    def get_cluster_records(self, cluster_id: int) -> List[SegmentRecord]:
        """Return all records belonging to a given cluster."""
        if self._cluster_labels is None:
            self.cluster()
        return [
            self._candidates[i].record
            for i, lbl in enumerate(self._cluster_labels)
            if int(lbl) == cluster_id
        ]

    # ── Promotion ────────────────────────────────────────────────────

    def promote(
        self,
        bank,
        observations_by_traj: Dict[str, list],
        llm_config=None,
    ) -> List[str]:
        """Attempt to promote qualifying clusters into real skills.

        This replaces the old ``materialize_new_skills()`` logic in pipeline.py
        with richer criteria: support + consistency + separability.

        Parameters
        ----------
        bank : SkillBankMVP
            The skill bank to add new skills to.
        observations_by_traj : dict
            Trajectory observations for Stage 3 processing.
        llm_config : LLMTeacherConfig, optional
            If provided, use LLM to suggest human-readable names.

        Returns
        -------
        list[str]
            IDs of newly created skills.
        """
        from skill_agents.stage3_mvp.run_stage3_mvp import (
            run_stage3_mvp,
            SegmentSpec,
        )
        from skill_agents.stage3_mvp.config import Stage3MVPConfig

        candidates = self.get_candidates()
        if not candidates:
            return []

        # Sort by (consistency * size) descending — prefer well-supported clusters
        candidates.sort(key=lambda c: -(c.consistency * c.size))

        created_ids: List[str] = []
        ts = int(time.time())

        for rank, summary in enumerate(candidates[:self.config.max_promotions_per_call]):
            if summary.size < self.config.min_cluster_size:
                continue
            if summary.consistency < self.config.min_consistency:
                continue

            cluster_recs = self.get_cluster_records(summary.cluster_id)
            if not cluster_recs:
                continue

            # Check distinctiveness against existing bank
            centroid_effects = summary.effect_centroid_add | summary.effect_centroid_del
            if not self._is_distinctive(centroid_effects, bank):
                continue

            new_id = f"S_new_{ts}_{rank}"
            for rec in cluster_recs:
                rec.skill_label = new_id

            specs = [
                SegmentSpec(
                    seg_id=rec.seg_id,
                    traj_id=rec.traj_id,
                    t_start=rec.t_start,
                    t_end=rec.t_end,
                    skill_label=new_id,
                    ui_events=list(rec.events) if getattr(rec, "events", None) else [],
                )
                for rec in cluster_recs
            ]

            s3_config = Stage3MVPConfig(
                min_instances_per_skill=max(1, self.config.min_cluster_size),
            )

            s3_summary = run_stage3_mvp(
                segments=specs,
                observations_by_traj=observations_by_traj,
                config=s3_config,
                bank=bank,
            )

            if new_id in s3_summary.skill_results:
                sr = s3_summary.skill_results[new_id]
                if sr.get("pass_rate", 0) >= self.config.min_pass_rate:
                    created_ids.append(new_id)
                    self.remove_promoted({rec.seg_id for rec in cluster_recs})

                    # Try to name the skill via LLM
                    contract = bank.get_contract(new_id)
                    if contract is not None and llm_config is not None:
                        self._try_name_skill(
                            contract, cluster_recs, observations_by_traj, llm_config,
                        )
                        bank.add_or_update(contract)
                else:
                    bank.remove(new_id)
                    for rec in cluster_recs:
                        rec.skill_label = "__NEW__"

        return created_ids

    def _is_distinctive(self, centroid_effects: Set[str], bank) -> bool:
        """Check candidate effects are sufficiently different from existing skills."""
        if not centroid_effects:
            return False

        for sid in bank.skill_ids:
            contract = bank.get_contract(sid)
            if contract is None:
                continue
            existing = (contract.eff_add or set()) | (contract.eff_del or set())
            if not existing:
                continue
            union = len(centroid_effects | existing)
            inter = len(centroid_effects & existing)
            jacc_dist = 1.0 - (inter / union) if union > 0 else 1.0
            if jacc_dist < self.config.min_distinctiveness:
                return False
        return True

    @staticmethod
    def _try_name_skill(
        contract: SkillEffectsContract,
        cluster_recs: List[SegmentRecord],
        observations_by_traj: Dict[str, list],
        llm_config,
    ) -> None:
        """Best-effort LLM naming for a newly promoted skill."""
        try:
            from skill_agents.infer_segmentation.llm_teacher import suggest_skill_name

            observation_slices = []
            for rec in cluster_recs[:3]:
                obs = observations_by_traj.get(rec.traj_id, [])
                if rec.t_start is not None and rec.t_end is not None:
                    sl = obs[rec.t_start: rec.t_end + 1]
                    if len(sl) > 0:
                        observation_slices.append(sl)

            if observation_slices:
                naming = suggest_skill_name(
                    observation_slices,
                    eff_add=list(contract.eff_add) or None,
                    eff_del=list(contract.eff_del) or None,
                    eff_event=list(contract.eff_event) or None,
                    config=llm_config,
                )
                if naming and naming.get("name"):
                    contract.name = naming["name"]
                    contract.description = naming.get("description")
        except Exception:
            pass  # naming is best-effort

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Save pool state to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "candidates": [
                {
                    "record": c.record.to_dict(),
                    "predecessor_skill": c.predecessor_skill,
                    "successor_skill": c.successor_skill,
                    "added_at": c.added_at,
                }
                for c in self._candidates
            ],
            "promoted_ids": sorted(self._promoted_ids),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, filepath: str) -> None:
        """Load pool state from JSON."""
        path = Path(filepath)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._candidates = []
        for entry in data.get("candidates", []):
            rec = SegmentRecord.from_dict(entry["record"])
            cand = NewCandidate(
                record=rec,
                predecessor_skill=entry.get("predecessor_skill"),
                successor_skill=entry.get("successor_skill"),
                added_at=entry.get("added_at", 0.0),
            )
            self._candidates.append(cand)
        self._promoted_ids = set(data.get("promoted_ids", []))
        self._cluster_labels = None

    def summary(self) -> Dict[str, Any]:
        """Compact summary for logging."""
        candidates = self.get_candidates()
        return {
            "pool_size": len(self._candidates),
            "n_clusters": len(self._cluster_summaries) if self._cluster_summaries else 0,
            "n_mature_clusters": len(candidates),
            "mature_clusters": [
                {
                    "id": c.cluster_id,
                    "size": c.size,
                    "consistency": round(c.consistency, 3),
                    "mean_duration": round(c.mean_duration, 1),
                    "sig": c.representative_sig,
                }
                for c in candidates[:5]
            ],
        }
