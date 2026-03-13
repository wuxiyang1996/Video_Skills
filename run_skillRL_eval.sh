#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=0 python cold_start/ALFWORLD-7B/generate_cold_start_lmgame_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-SFT --episodes 30 &

CUDA_VISIBLE_DEVICES=1 python cold_start/ALFWORLD-7B/generate_cold_start_evolver_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-SFT --episodes 30 &

CUDA_VISIBLE_DEVICES=2 python cold_start/ALFWORLD-7B/generate_cold_start_orak_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-SFT --games super_mario --episodes 30 &

CUDA_VISIBLE_DEVICES=3 python cold_start/ALFWORLD-7B/generate_cold_start_pokemon_red_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-SFT --episodes 30 --verbose &

wait
echo "All four jobs finished."
