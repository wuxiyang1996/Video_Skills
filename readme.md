# Enhance Agentic Decision-making in Multiple-player long-horizon games with unsupervised and synthetic experiences

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/wuxiyang1996/Game-AI-Agent)

## Overview

This repository provides a framework for enhancing agentic decision-making in multi-player, long-horizon games through unsupervised and synthetic experience generation. The framework integrates with multiple game environments and supports both training-free (RAG-based) and trainable (RL-based) agent architectures.

## Quick Links

- **📦 Environment Wrappers**: [env/](env/) - Natural language wrappers and evaluation scripts for game environments
  - [Overcooked AI](env/env_wrappers/overcooked_nl_wrapper.py) - Cooperative cooking game wrapper
    - Source: [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) by HumanCompatibleAI
  - [Avalon](env/env_wrappers/avalon_nl_wrapper.py) - Hidden-role deduction game wrapper
    - Source: [AgentEvolver Games](https://github.com/modelscope/AgentEvolver/blob/main/games/README.md) by ModelScope
  - [Diplomacy](env/env_wrappers/diplomacy_nl_wrapper.py) - Strategic negotiation game wrapper
    - Source: [AgentEvolver Games](https://github.com/modelscope/AgentEvolver/blob/main/games/README.md) by ModelScope
  - [Evaluation Scripts](env/README.md) - Test scripts for running agents in these environments

- **🔍 RAG & Embeddings**: [rag/](rag/) - Embedding models for experience retrieval
  - Text (RAG) embedding: default [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
  - Multimodal embedding: default [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
  - Configurable via `RAG_EMBEDDING_MODEL` and `MULTIMODAL_EMBEDDING_MODEL` or constructor args; see [rag/README.md](rag/README.md)

- **✂️ Stage 1 Boundary Proposal**: [skill_agents/boundary_proposal/](skill_agents/boundary_proposal/) - High-recall candidate cut points for trajectory segmentation
  - Integrated with `Episode` / `SubTask_Experience` data structures and per-env signal extractors (Overcooked, Avalon, Diplomacy)
  - Optional RAG-embedding change-point detection via `TextEmbedder`
  - `segment_episode()` — full pipeline: Episode → SubTask_Experience segments
  - `annotate_episode_boundaries()` — mark boundaries for `separate_into_sub_episodes()`
  - See [skill_agents/boundary_proposal/README.md](skill_agents/boundary_proposal/README.md)

- **🔗 Repository**: [GitHub - Game-AI-Agent](https://github.com/wuxiyang1996/Game-AI-Agent)

## Dependencies

This framework integrates with the following game environments:

- **Overcooked AI**: Uses [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) - A benchmark environment for fully cooperative human-AI performance, based on the popular video game Overcooked.

- **Avalon & Diplomacy**: Uses [AgentEvolver Games](https://github.com/modelscope/AgentEvolver/blob/main/games/README.md) - A unified arena for interaction, evaluation, and training of AI agents in social reasoning games (Avalon: hidden-role deduction, Diplomacy: strategic negotiation).

## Introduction

The intention of this readme file is to provide an outline of each module serving for different purposes, and provide some initial instructions for vibe coding, also for the ease of integration and debugging.

The instruction prompt includes the function definition and the ToDo list, including all the functions under a class serving for different purposes.

#####################

# Data Strucutre and Pre-processing code
The intention for this part is to clean up the initial experiences from external sources into uniform format, also provide some necessary labelings to fill up the blank field.

ToDo:
Need to include the following data structure:
1. Experience
2. Episode, composed by exprience, time length limited
3. Expereience buffer, limited in its size

## Experience
Same as the normal Reinforcement Learning setting, we define the experience format are given by

State, action, intention, reward, next state, done

Long-term Goal, short-term goal, Synthesis label, experience quality score, etc. are intentionally leaving blank to be filled.
May leave another field for experience value for priorized replay
Need to also include a summary field to allow RAG and embeddding models.

Please note that depending on the variance of the environment involved, not all information are provided.

The actual state taken by LLM agents are given by summary of the state, also serving as the context for RAG models to retrieve

## Epiosde 
Where each episode is composed by such kind of experiences, the final reward, length and overall comments are also provided

ToDo: 
1. We need a experience buffer with in and out policy models, that evaluates the relative quality and relevance to the proposed latent state (intentions / sub-term goals) and the current state given. The experience data structure should have push and pop out function to add/remove new experiences
2. We need some helper function for each environment that covert the experience sequences gathered in each environment into the format we are expecting, also automatically fill the blank fields within the raw experience. Note, in this step, some sample experiences must be provided.
3. Leaving a place to allow experiece quality evaluation and priorized experience replay. If doing so, should also add a auto-ranking function within the experience buffer to compute this value
4. Add a experience summary code if summary is not included, add a experience embeddding code (to allow RAG and experience query).


#####################

# Sub-task Decomposition

This is for experience pre-processing coming from exteral source if unlabeled, or rollouts from sub-optimal policies.

Suppose that we have the experiences processed into the intended format, the intention of this step is to decompose the entire experience trajectories into sub-tasks and sub-trajectories, i.e., the input trajectory is unlabeled without any annotation over the entire trajectory, expected output should be sub-task labeling over the entire trajectory.


ToDo: 
1. Experience clean-up code, or some helper function, convert the unformulated experiences into formulated one. In such a function, we need a intention inference goal to add pesudo intention labeled over the trajectory. This part can come from the previous part.
2. Sub-task decomposition code: Read the unlabeled full trajectory from exteral source, use LLM to perform unsupervised learning and to label sub-tasks over some specific tasks, add some constraints over the labeling, including (1) Maximum allow sub-task length (2) Provide some candidate options to choose from, coming from the intentions (ego or opponents), given the current state
3. Training code: It seems like there's no direct task labeling signal for training LLM (if necessary), one potential solution is to send the experience labeling over rollout trajectories (coming from the real-world demostrations) with reward signals (sub-task assignment, task-related rewards).

As for the tasks, it will be nice to make it formulated, i.e. constrain the task type into some finite categories if possible.

If necessary, the labeling or training process can also be in-context learning that generate the output based on the demostrations retrieved.

#####################

# Reward Labeling (for tasks)

(I am not sure if this could work, TBD)

The general idea for this part is to compute the similaity between the next state of one experience, to encourage the agent to achieve (exploit) the maximum similaty to achieve the final sub-task/final task state

Well, the embedding state made by the LLM should be slightly diffrent from the one used by RAG.

Notablbly, such a function should be done with CoT, to improve its plausiblity.

ToDo:
1. Call the summary generation function to get the summary of the state
2. Compute the reward score between the target state and the next state of the current experience queried. Note that, the reward score is not the similarity score like the RAG, as the initial state, though low in terms of similarity score, still should induce high reward when contributing to the final goal state.

(Why not think about using the increase/decrease value of the reward in RAG state, not the actual value)

#####################

# Completion Validator

This function intends to verify if the task / sub-task completed, by deploying the function involved in the section above. 

Notablbly, such a function should be done with CoT, to improve its plausiblity. Also, casuality may need to be involved here.

ToDo:
1. The most essential thing needs to be done is the LLM function that loads the state and determine the completion level of the current task. Such a function is different from the one in the RAG section, or reward computation.
e.g. (1) High rewarding action in early stage -> Low completion score, high reward
(2) Low rewarding action in late stage -> High completion score, low reward
2. Involve another function, reading the initial state (not the summary), and the final state, deteremine how many steps left before reaching the final state? Could use stored experience coming from RAG.

#####################

# Experience Retreival
Use RAG, to get the experience from the buffer when generating new experience, or using training-free mode to make decisions when perform decision-making using in-context learning.

ToDo:
1. Deploy external RAG model, to generate the embedding of the experiences
2. Query the experience which is the most relevant to the current state and intentions, given the similiaty score from the RAG model
3. Update the RAG model using the idea similar to constrastive learning (or other RAG model training techniques) 

#####################

# Experience Synthesis
Counterfacual action generation, given the state, and current action, output what if an alternative valid actions taken, what will be the outcome

The generated experience needs validation, such a validation is done by CoT (Details included in Completion Validator).

ToDo:
1. Experience synthesis function, taking the current state summary, the intentions, the historial states, to (1) find out the available actions within the state; (2) If necessary, modify the sub task assignment, but need to make it satisfy some constraints first (3) infer the next state and the reward given the next state.
2. Need the extra validation for the generation qualities, deteremine whether the generated experiences aligned with the reality, taking LLM-as-a-Judge, mostly is a optional function that outputs True/False.
3. Here, we could allow multiple kind of rollouts: Single state-action experience synthesis, multi-step sub-task level experience synthesis. Allowing the validation code works in these situations. Need to consider the diversity of the experience generated here.
4. Training code, using SFT to enhance the understanding of the experience sythesis, also improve the plausiblity of experiences.

#####################

# Decision-making agent design

This part is to design an agent to decision-making that try to solve the task involved. Either using trainable (with Reinforcement Learning) or training-free matter that uses in-context learning by RAG.

## Training Free Agent
This kind of agent uses RAG to query experience that are most relevant to the currnet situation and intentions from the experience buffer, using them as in-context learing to assist decision-making. Such a agent is using GPT/Gemini/Claude model as a backbone.

## Trainable Agent
This kind of agent gathers experience via interaction and synthesis, updates the parameters to improve its decision-making abilities via reinforcement learning.

ToDo:
We denote that the agent we proposed for decision-making should have the following modules
1. Summarize the state (GPT: Training-free, Qwen: Trainable)
2. Find out reachable tasks from the current state, see if needed to update the current sub-task or intention.
3. (Training-free) Integrate a RAG module within the agent to query the relevant experience from the buffer.
4. Using the current state (summary), historal experience (in this episode), relevant state (only for Training-free), to generate new actions.

Note, the counterfactual action generation module should use the same structure, but the gradient should not be updated from there.

#####################

# Reward Design for agents
This part is for agents, we use constrastive learning to train the RAG and use SFT to train teh experience synthesis.

We adpot the idea of GDPO that we normalize each kind of reward under each category. Rewards that we want to involve in this system include:
1. Task Completion Reward, including the winning condition given by the game, or just trying to servive as long as possible, coming from the environment.
2. Sub-task Completion Reward, adopt from the reward like occuptation, kill, etc.
3. Format reward.

Please note that the first two rewards are env-related. Also, a time-senstive-discount is applied onto that to apply penality over the episode length as well.

#####################

# Trainer Code
The training code intends to use the experience to train agents, allow it to (i) take better actions (2) get better summary of the environment (3) Find and get the decion

ToDo:
1. Trainer Code, need to adopt PPO code, GRPO code base and GDPO code base, also, need to include LoRA for fast fine-tuning and should be based on R1 codebases for the ease of debugging and modification.
2. We must need a priorized experience replay module to improve the training performance later.
3. All trainers should allow LoRA options when training.


####################

## Framework
Agent for decision-making, frozen, fully depending on the memory retrieval


### World model for experience synthesis

Take it as a image editing task, taking the skills and current state as the input, generate teh intended state

Longcat 
https://huggingface.co/meituan-longcat/LongCat-Image-Edit

Qwen-Edit
https://huggingface.co/Qwen/Qwen-Image-Edit-2511

Bagel
https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
 