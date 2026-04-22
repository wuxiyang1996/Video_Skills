# 99_meta — Tracking Documents

> **Purpose:** Hold cross-cutting **tracking** documents that are *about* the plans rather than part of the system design itself. The numbering uses `99_` so it sorts last and is visibly separate from the design layers `00` – `07`.

---

## Files in this folder

| File | What it is | When you read it |
|------|------------|------------------|
| [`plan_docs_implementation_checklist.md`](plan_docs_implementation_checklist.md) | A per-plan checklist of what is **done** vs **open**, organized by plan file and edited as the plans evolve. The grounding-related items (§3 entire checklist; §2 entity schema) are closed; the reasoning + skills + memory-lifecycle items are still being checked off. | When triaging what to work on next; when reviewing a PR that touches a plan; when you need to see whether a particular requirement has already been addressed elsewhere. |

---

## What lives here vs what lives in the layer folders

- A **plan** ("how the controller should work") goes in `03_controller/`.
- A **checklist** ("what is still missing across all plans") goes here in `99_meta/`.
- A **status update on a single subsystem** belongs in that subsystem's folder, not here.

If a third meta-document is added later (e.g. a release log or a contributor guide for editing plans), it goes here.
