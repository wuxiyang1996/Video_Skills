# Inference

Run the **decision agent** ([decision_agents/](../decision_agents/)) and store rollouts in the **data_structure** format ([data_structure/experience.py](../data_structure/experience.py)): `Experience` list and `Episode`.

## Usage

```python
from inference import run_inference, rollout_to_episode
from data_structure.experience import Episode_Buffer, Experience_Replay_Buffer

# Run one episode and get an Episode (list of Experience)
episode = run_inference(
    env,
    task="Complete level 1",
    max_steps=500,
    verbose=True,
)

# Optional: add to buffers and/or append to a JSONL file
ep_buffer = Episode_Buffer(buffer_size=100)
exp_buffer = Experience_Replay_Buffer(buffer_size=10_000)
episode = run_inference(
    env,
    task="Complete level 1",
    episode_buffer=ep_buffer,
    experience_buffer=exp_buffer,
    save_path="rollouts/episodes.jsonl",
    verbose=True,
)
```

## Converting an existing rollout

If you already have a `run_episode_vlm_agent` result (e.g. from the trainer or a custom loop):

```python
from decision_agents import run_episode_vlm_agent
from inference import rollout_to_episode

rollout = run_episode_vlm_agent(env, max_steps=500)
episode = rollout_to_episode(rollout, task="My task")
# episode.experiences: list of Experience (state, action, reward, next_state, done, ...)
# episode.to_dict(): for JSON save/load
```

## Storage format

- **Episode**: `experiences` (list of `Experience`), `task`, `outcome`, etc. See `Episode.to_dict()`.
- **Experience**: `state`, `action`, `reward`, `next_state`, `done`, optional `intentions`, `tasks`, `sub_tasks`, `summary`, `idx`, etc.
- **save_path**: If provided, each episode is appended as one JSON object per line (JSONL), so multiple runs can append to the same file.

## Dependencies

- `decision_agents`: `run_episode_vlm_agent`, `VLMDecisionAgent`
- `data_structure.experience`: `Experience`, `Episode`, optionally `Episode_Buffer`, `Experience_Replay_Buffer`
