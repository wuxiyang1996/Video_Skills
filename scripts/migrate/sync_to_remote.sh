#!/usr/bin/env bash
# Sync the training/evaluation bundle (no raw videos) to a rented machine.  Usage: sync_to_remote.sh user@host:/workspace
# Sizes: VH data 4.1G, Qwen3.5-9B 24G, VH test catalogs (gen1+gen2 narratives; stages regenerable) ~6G, train catalog 1.5G,
# reader data/adapters ~1G, CG/VRBench catalogs ~3G, keys. Run from the cluster login node.
set -euo pipefail; DEST=${1:?user@host:/path}
R() { rsync -aH --info=progress2 --partial "$@"; }
R --exclude '.git' --exclude 'dataset_clip_wrapper/output' --exclude '.venv*' /fs/gamma-projects/vlm-robot/Video_Skills/ "$DEST/Video_Skills/"
R /fs/gamma-projects/vlm-robot/datasets/Video-Holmes/Benchmark/videos_cropped /fs/gamma-projects/vlm-robot/datasets/Video-Holmes/Benchmark/test_Video-Holmes.json /fs/gamma-projects/vlm-robot/datasets/Video-Holmes/Benchmark/train_Video-Holmes.json /fs/gamma-projects/vlm-robot/datasets/Video-Holmes/Benchmark/annotations /fs/gamma-projects/vlm-robot/datasets/Video-Holmes/Benchmark/annotation_training "$DEST/datasets/Video-Holmes/Benchmark/"
R /fs/gamma-projects/vlm-robot/datasets/VRBench/VRBench_eval.jsonl "$DEST/datasets/VRBench/"
R /fs/gamma-projects/vlm-robot/datasets/CG-Bench/cgbench.json /fs/gamma-projects/vlm-robot/datasets/CG-Bench/cgbench_mini.json "$DEST/datasets/CG-Bench/"
R /fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache/hub/models--Qwen--Qwen3.5-9B "$DEST/hf_cache/hub/"
S=/fs/nexus-scratch/wuxiyang
R $S/vh_l1_levers/narr_px_plus_full $S/vh_l1_levers/narr_px_plus_full_rep $S/vh_l1_levers/narr_px_plus $S/vh_l1_levers/narr_px $S/vh_l1_levers/asr $S/vh_l1_levers/cg_narr_px_plus $S/vh_l1_levers/cg_asr $S/vh_l1_levers/all_ids_1837.txt $S/vh_l1_levers/cg_ids_237.txt $S/vh_l1_levers/cg_v1_index_237.json "$DEST/scratch/vh_l1_levers/"
R $S/vh_train_l1/narr_px_plus $S/vh_train_l1/asr $S/vh_train_l1/train_ids.txt "$DEST/scratch/vh_train_l1/"
R $S/reader_train/sft_v1.jsonl $S/reader_train/sft_v1_p50.jsonl $S/reader_train/teacher_s0.rollouts.jsonl $S/reader_train/teacher_s1.rollouts.jsonl $S/reader_train/teacher_s2.rollouts.jsonl $S/reader_train/sft_v2 $S/reader_train/sft_v2b "$DEST/scratch/reader_train/"
R $S/vrbench_pilot_v1/derived_pilot60 $S/vrbench_pilot_v1/measure_pilot60 $S/vrbench_pilot_v1/heldout60/derived $S/vrbench_pilot_v1/heldout60/measure "$DEST/scratch/vrbench_pilot_v1/"
R /fs/gamma-projects/vlm-robot/Video_Skills/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/full_vh_mm /fs/gamma-projects/vlm-robot/Video_Skills/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/cg_qa /fs/gamma-projects/vlm-robot/Video_Skills/dataset_clip_wrapper/output/vh_full_questions_v1 "$DEST/Video_Skills/dataset_clip_wrapper/output/"
echo "keys.py is NOT synced: copy it out of band to $DEST/keys.py and chmod 600"
