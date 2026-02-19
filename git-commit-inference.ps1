# Stage inference module and readme; commit.
git add inference/ readme.md
git commit -m "feat: add inference module and document usage in readme

- inference/: run decision agent and store rollouts in data_structure format
  - run_inference(env, task=..., episode_buffer=..., save_path=...)
  - rollout_to_episode(rollout, task=...) for existing run_episode_vlm_agent result
- readme: Inference quick link; Inference usage section (run, buffers, JSONL, convert)"
