# 1. LMGame-Bench (2048, Sokoban, Tetris, Candy Crush)
python cold_start/ALFWORLD-7B/generate_cold_start_lmgame_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-SFT --episodes 20

<!-- python cold_start/ALFWORLD-7B/generate_cold_start_lmgame_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-RL --episodes 20 -->

# 2. AgentEvolver (Avalon, Diplomacy)
python cold_start/ALFWORLD-7B/generate_cold_start_evolver_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-SFT --episodes 20

<!-- python cold_start/ALFWORLD-7B/generate_cold_start_evolver_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-RL --episodes 20 -->

# 3. Orak (Super Mario)
python cold_start/ALFWORLD-7B/generate_cold_start_orak_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-SFT --games super_mario --episodes 1

<!-- python cold_start/ALFWORLD-7B/generate_cold_start_orak_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-RL --games super_mario --episodes 20 -->

# 4. Pokemon Red
python cold_start/ALFWORLD-7B/generate_cold_start_pokemon_red_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-SFT --episodes 3 --verbose

<!-- python cold_start/ALFWORLD-7B/generate_cold_start_pokemon_red_alfworld7b.py \
  --model_path Jianwen/Alfworld-7B-RL --episodes 3 --verbose -->