# This file is to define the data structure for the experience, which is used to store the experience of the agent.
# The learned/rollout data are stored in the differetn data structure, including: 
# 1. Experience: Single state-action-next_state-reward-done tuple.
# 2. SubTask_Experience (strategy or tool): A collection of experiences, accomplishing some stage-wise tasks over a few steps.
# 3. Episode: A collection of experiences, which is used to store the experience over the entire game rollout of the agent.
# Also we need to include experience buffers and tool buffers to store the experience and tool/strategy for the agent. The storage optimization is for later tasks.
# For each experience and sub-task experience, we need to include the evalution interface for external evaluation functions.

# Please note that this code only include the data structure code, data management and data processing are for other files.

from __future__ import annotations  # Enable postponed evaluation of annotations

import uuid
from typing import List, Optional, TYPE_CHECKING
from API_func import ask_model
import random
import json
from pathlib import Path
# Note: helper.py imports are commented out to avoid circular import
# from data_structure.helper import *

# The data structure for the experience, which is used to store the experience of the agent.
# TODO: Multi-modality
class Experience:
    def __init__(self, state, action, reward, next_state, done, intentions=None, tasks=None, sub_tasks=None):
        # Required fields, must to be filled in the experience generation process.
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

        # The done is just a label marking the end of an episode.
        self.done = done
        # The sub-task done is for marking whether the current sub-task is completed.
        self.sub_task_done = None

        # Optional fields, could be filled in the experience generation process.
        # Index of the experience in the episode.
        self.idx = None

        # Optional, but required in our design. Could be remove in the baselines.
        # Involving the intentions and tasks generation process, also the sub-task labelings.
        # Please note that in each game, we assume that there might be a long-horiztion task for the agent to accomplish.
        self.intentions = intentions
        self.tasks = tasks # Long-term tasks, which is the goal of the agent.
        self.sub_tasks = sub_tasks # Short-term tasks, represening the current strategy of the agent.
        
        # Summarization of the experience.
        # Essential for experience query and retrieval.
        self.summary = None
        # Summary of the state of the experience, used for quick retrieval and query.
        self.summary_state = None

        # Per-step reward breakdown (r_env, r_follow, r_cost, r_total).
        # Populated by the VLM decision agent's reward tool.
        self.reward_details: Optional[dict] = None

        # Action classification: "primitive", "QUERY_MEM", "QUERY_SKILL", "CALL_SKILL".
        # Used by the trainer for reward cost computation and action-type metrics.
        self.action_type: Optional[str] = None

    # Generate the summary of the experience.
    # The intention of this function is to generate the summmary of the current state and 
    def generate_summary(self):
        # Generate the summary of the experience.
        # We mainly focus on the state, action and next state, but exclude the done, which is just a label marking the end of an episode.
        prompt = (
            f"Generate the summary of the experience. "
            f"Focusing on the current state over the ego agents' state and other agents' states that prompts the agent to take the actions."
            f"Capture the key points of the state but make it very concise."
            f"The state is {self.state}, the action is {self.action}, the next state is {self.next_state}."
            f"Please include the following information: "
            f"1. The summary of the state, including the ego agents' state and other agents' states."
            f"2. The summary of the action. Make it very concise."
            f"3. The summary of the next state. Focusing on the outcome of the ego agents' action"
        )
        self.summary = ask_model(prompt)
        return self.summary

    def generate_summary_state(self):
        # Generate the summary of the state.
        prompt = (
            f"Generate the summary of the state. "
            f"Focusing on the current state over the ego agents' state and other agents' states that prompts the agent to take the actions, but exclude the action taken"
            f"Capture the key points of the state but make it very concise."
            f"The state is {self.state}. "
        )
        self.summary_state = ask_model(prompt)
        return self.summary_state

    # Helper to generate intentions for this experience (for experience labeling).
    # We should have intentions for our own agents.
    def generate_intentions(self, history: List[Experience]):
        prev_summary_list = [exp.summary for exp in history]
        prompt = (
            f"Generate the intentions for the experience, while the intentions marks the motivation for agent to take the action and achieve the sub-task. "
            f"Also, the historical state over the last few steps should be considered. "
            f"The current state is {self.state}, the current action is {self.action}, the current next state is {self.next_state}, the current sub-task is {self.sub_tasks}. "
            f"The history is {prev_summary_list}. Given in the time-wise order."
        )
        self.intentions = ask_model(prompt)
        return self.intentions

    # Helper function to generate the intentions or summaries if not provided.
    # For now only use summary, not summary_state.
    def initialize_intentions_and_summary(self, history: Optional[List[Experience]] = None):
        if self.intentions is None:
            self.generate_intentions(history)
        if self.summary is None:
            self.generate_summary()
        return self.intentions, self.summary

    def to_dict(self):
        d = {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "intentions": self.intentions,
            "tasks": self.tasks,
            "sub_tasks": self.sub_tasks,
            "summary": self.summary,
            "summary_state": self.summary_state,
            "idx": self.idx,
        }
        if self.reward_details is not None:
            d["reward_details"] = self.reward_details
        if self.action_type is not None:
            d["action_type"] = self.action_type
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "Experience":
        """Construct an Experience from a dictionary."""
        exp = cls(
            state=d["state"],
            action=d["action"],
            reward=d["reward"],
            next_state=d["next_state"],
            done=d["done"],
            intentions=d.get("intentions"),
            tasks=d.get("tasks"),
            sub_tasks=d.get("sub_tasks"),
        )
        exp.summary = d.get("summary")
        exp.summary_state = d.get("summary_state")
        exp.idx = d.get("idx")
        exp.reward_details = d.get("reward_details")
        exp.action_type = d.get("action_type")
        return exp


# Episode is the collection of experiences, which is used to store the experience of the agent.
# Please include the intention and task generation process in the episode.
# The episode initially from the rollout, leaving for fruther process and push into the experience replay buffer.
class Episode:
    def __init__(
        self,
        experiences: List[Experience],
        task: str,
        metadata: Optional[dict] = None,
        episode_id: Optional[str] = None,
        env_name: Optional[str] = None,
        game_name: Optional[str] = None,
    ):
        self.experiences = experiences

        # Unique identifier for this episode (auto-generated if not provided).
        self.episode_id: str = episode_id or str(uuid.uuid4())

        # Platform / wrapper name (e.g. "gamingagent", "videogamebench_dos", "overcooked").
        self.env_name: Optional[str] = env_name

        # Specific game within the platform (e.g. "sokoban", "2048", "doom2", "kirby").
        self.game_name: Optional[str] = game_name

        # The summary of the episode.
        self.summary = None

        # The tasks of the episode, which is unique for each episode.
        self.task = task

        # The outcome of the episode.
        self.outcome = None

        # Arbitrary metadata (cumulative_reward, agent_state snapshot, etc.).
        self.metadata: Optional[dict] = metadata

    def get_reward(self):
        """Sum of raw environment rewards (r_env) across all experiences."""
        return sum(experience.reward for experience in self.experiences)

    def get_total_reward(self):
        """Sum of shaped total rewards (r_total) from reward_details.

        Falls back to r_env (self.reward) when reward_details is absent.
        """
        total = 0.0
        for exp in self.experiences:
            if exp.reward_details and "r_total" in exp.reward_details:
                total += exp.reward_details["r_total"]
            else:
                total += exp.reward
        return total

    def get_length(self):
        return len(self.experiences)

    def set_outcome(self):
        # Set the outcome of the episode.
        self.outcome = self.experiences[-1].done
        return self.outcome
    
    def generate_summary(self):
        # Generate the summary of the episode.
        prompt = (
            f"Generate the summary of the episode. "
            f"The episode is {self.experiences}, the task is {self.task}, the outcome is {self.outcome}. "
        )
        self.summary = ask_model(prompt)
        return self.summary

    # This is a function to separate the episode into sub-episodes.
    # The outcome_length is the number of steps to look ahead to evaluate the outcome of the sub-task, is a hyper-parameter.
    def separate_into_sub_episodes(self, outcome_length: int = 5):
        all_sub_tasks = []
        # this is to store the index of the starting experience of the sub-task in the episode.
        sub_task_idx = []
        sub_episodes = []
        curr_sub_task = None

        # Find out the sub-task indices in the episode.
        for experience in self.experiences:
            if experience.sub_tasks is not None and experience.sub_tasks not in all_sub_tasks:
                all_sub_tasks.append(experience.sub_tasks)
                sub_task_idx.append(experience.idx)
                curr_sub_task = experience.sub_tasks
        
        sub_task_idx.append(len(self.experiences) - 1)

        # Segment the episode into sub-episodes given the idx.
        for i in range(len(sub_task_idx) - 1):
            curr_idx = sub_task_idx[i]
            next_idx = sub_task_idx[i + 1]
            if next_idx + outcome_length <= len(self.experiences):
                sub_episodes.append(SubTask_Experience(curr_sub_task, self.task, 
                self.experiences[curr_idx:next_idx], self.experiences[next_idx:next_idx+outcome_length]))
            else:
                sub_episodes.append(SubTask_Experience(curr_sub_task, self.task, 
                self.experiences[curr_idx:next_idx]))
        return sub_episodes

    def to_dict(self):
        d = {
            "episode_id": self.episode_id,
            "env_name": self.env_name,
            "game_name": self.game_name,
            "experiences": [exp.to_dict() for exp in self.experiences],
            "task": self.task,
            "outcome": self.outcome,
            "summary": self.summary,
        }
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        """Construct an Episode from a dictionary."""
        experiences = [Experience.from_dict(exp) for exp in d["experiences"]]
        ep = cls(
            experiences=experiences,
            task=d["task"],
            metadata=d.get("metadata"),
            episode_id=d.get("episode_id"),
            env_name=d.get("env_name"),
            game_name=d.get("game_name"),
        )
        ep.outcome = d.get("outcome")
        ep.summary = d.get("summary")
        return ep

# The intention of this class is to store the experience of the agent for each sub-task.
# While the sub-tasks could be taken as a part of the entire episode.
# Denote that this data structure is used for labeling strategies or tools to complete the sub-task, within a limited length.
# Please note that this is for training purposes, only update after the current rollout is completed.
class SubTask_Experience:
    def __init__(self, sub_task: str, final_goal: str, experiences: List[Experience], outcome:Optional[List[Experience]] = None, seg_id: Optional[str] = None):
        # What's this strategy or tool used for.
        self.sub_task = sub_task
        self.final_goal = final_goal
        # Contents of the sub-task experience.
        self.sub_task_experience = experiences

        # Outcome of the sub-task.
        self.outcome_experiences = outcome
        
        # Link to the corresponding SegmentRecord from the skill pipeline.
        self.seg_id: Optional[str] = seg_id

        # The summary for query this strategy or tool.
        self.summary = None
        # The summary of the outcome of the sub-task. 
        self.outcome_summary = None

        # The length of the sub-task experience.
        self.length = len(experiences)

        # The cumulative reward of the sub-task experience. For the ease to quick retrieval and query.
        self.cumulative_reward = sum(exp.reward for exp in experiences)

    # Helper to generate intentions for this experience (for experience labeling).
    # We should have intentions for our own agents.
    def generate_summary(self):
        prev_summary_list = [exp.summary for exp in self.sub_task_experience]
        # Handle case when sub_task is None
        if self.sub_task is None:
            prompt = (
                f"Summarize the agent's strategy and motivation to achieve the final goal '{self.final_goal}'. "
                f"Experience history (time-ordered): {prev_summary_list}"
            )
        else:
            prompt = (
                f"Summarize why the agent chose sub-task '{self.sub_task}' to achieve the final goal '{self.final_goal}'. "
                f"Include the motivation and strategy. "
                f"Experience history (time-ordered): {prev_summary_list}"
            )
        self.summary = ask_model(prompt)
        return self.summary

    # The ideal outcome summary given the subsequences     
    def generate_outcome_summary(self):
        prev_outcome_summary_list = [exp.outcome_summary for exp in self.outcome_experiences]
        # Handle case when sub_task is None
        if self.sub_task is None:
            prompt = (
                f"Evaluate whether the agent's actions contributed to the final goal '{self.final_goal}'. "
                f"Consider if they induced rewarding actions in subsequent steps. "
                f"Subsequent outcomes (time-ordered): {prev_outcome_summary_list}"
            )
        else:
            prompt = (
                f"Evaluate whether completing the sub-task '{self.sub_task}' contributed to the final goal '{self.final_goal}'. "
                f"Consider if it induced rewarding actions in the subsequent steps. "
                f"Subsequent outcomes (time-ordered): {prev_outcome_summary_list}"
            )
        self.outcome_summary = ask_model(prompt)
        return self.outcome_summary

    # Only being used when labeling trajectory from external sources.
    def sub_task_labeling(self):
        # The summary of the sub-task experience.
        prev_summary_list = [exp.summary for exp in self.sub_task_experience]

        if self.outcome_experiences is not None:
            # The outcome summary of the sub-task experience.
            prev_outcome_summary_list = [exp.outcome_summary for exp in self.outcome_experiences]
            prompt = (
                f"Create a concise one-sentence label for this sub-task that describes the strategy used. "
                f"Sub-task summary: {prev_summary_list}. "
                f"Outcome summary: {prev_outcome_summary_list}"
            )
        else:
            prompt = (
                f"Create a concise one-sentence label for this sub-task that describes the strategy used. "
                f"Sub-task summary: {prev_summary_list}. "
            )
        self.sub_task_label = ask_model(prompt)
        return self.sub_task_label

    # Initialize the sub-task experience.
    def initialize_sub_task_experience(self):
        # Initialize the summary of the sub-task experience.
        if self.summary is None:
            self.generate_summary()
        
        # Initialize the outcome summary of the sub-task experience.
        if self.outcome_summary is None:
            self.generate_outcome_summary()

        # Initialize the sub-task label.
        if self.sub_task_label is None:
            self.sub_task_labeling()
        return self.summary, self.outcome_summary, self.sub_task_label
    

    def to_dict(self):
        d = {
            "sub_task": self.sub_task,
            "final_goal": self.final_goal,
            "sub_task_experience": [exp.to_dict() for exp in self.sub_task_experience],
            "seg_id": self.seg_id,
        }
        if self.outcome_experiences is not None:
            d["outcome_experiences"] = [exp.to_dict() for exp in self.outcome_experiences]
        else:
            d["outcome_experiences"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SubTask_Experience":
        """Construct a SubTask_Experience from a dictionary."""
        sub_task_exps = [Experience.from_dict(exp) for exp in d["sub_task_experience"]]
        outcome_exps = None
        if d.get("outcome_experiences"):
            outcome_exps = [Experience.from_dict(exp) for exp in d["outcome_experiences"]]
        return cls(
            sub_task=d["sub_task"],
            final_goal=d["final_goal"],
            experiences=sub_task_exps,
            outcome=outcome_exps,
            seg_id=d.get("seg_id"),
        )

# Experience Replay Buffer
# The intention of this class is to store the experience of the agent for the experience replay.
# One thing: Need to determine whether to include the experince or the episode.
#TODO: Prioritize the experience pop-out and selection for the episode. 
class Experience_Replay_Buffer:
    def __init__(self, buffer_size: int):
        self.buffer = []
        self.buffer_size = buffer_size

    # Add experience(s) to the buffer.
    # Enforces buffer size limit using FIFO (First In First Out) policy.
    # Supports both single experience and list of experiences (episode or strategy-level).
    def add_experience(self, experience):
        # Handle both single experience and list of experiences
        if isinstance(experience, list):
            # Add multiple experiences at once
            self.buffer.extend(experience)
        elif isinstance(experience, Episode):
            # Add all experiences from an episode
            self.buffer.extend(experience.experiences)
        else:
            # Add single experience
            self.buffer.append(experience)
        
        # Remove oldest experiences if buffer exceeds maximum size
        if len(self.buffer) > self.buffer_size:
            overflow = len(self.buffer) - self.buffer_size
            self.buffer = self.buffer[overflow:]

    def add_experiences(self, experiences: List[Experience]):
        """Add multiple experiences at once (convenience method)."""
        self.add_experience(experiences)

    def get_experience_summary(self, query: str):
        return [experience.summary for experience in self.buffer
                if experience.summary and query in experience.summary]

    def sample_experience(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# Episode Buffer
# The intention of this class is to store complete episodes for experience replay and analysis.
class Episode_Buffer:
    def __init__(self, buffer_size: int):
        self.buffer = []
        self.buffer_size = buffer_size

    # Add episode(s) to the buffer.
    # Enforces buffer size limit using FIFO (First In First Out) policy.
    # Supports both single episode and list of episodes.
    def add_episode(self, episode):
        # Handle both single episode and list of episodes
        if isinstance(episode, list):
            # Add multiple episodes at once
            self.buffer.extend(episode)
        elif isinstance(episode, Episode):
            # Add single episode
            self.buffer.append(episode)
        else:
            raise TypeError(f"Expected Episode or list of Episodes, got {type(episode)}")
        
        # Remove oldest episodes if buffer exceeds maximum size
        if len(self.buffer) > self.buffer_size:
            overflow = len(self.buffer) - self.buffer_size
            self.buffer = self.buffer[overflow:]

    def add_episodes(self, episodes: List[Episode]):
        """Add multiple episodes at once (convenience method)."""
        self.add_episode(episodes)

    def sample_episode(self, batch_size: int):
        """Sample random episodes from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    # Return all the episode summaries for query and RAG.
    def get_episode_summary(self, query: str):
        return [episode.summary for episode in self.buffer if episode.summary and query in episode.summary]
    
    def __len__(self):
        """Return the number of episodes in the buffer."""
        return len(self.buffer)

    # Convert the episode buffer to a dictionary.
    def to_dict(self):
        return {
            "episodes": [episode.to_dict() for episode in self.buffer],
        }
    
    # Convert the dictionary to an episode buffer.
    def from_dict(self, d: dict):
        self.buffer = [Episode.from_dict(ep) for ep in d["episodes"]]
        return self
    
    # Save episode buffer to JSON file.
    def save_to_json(self, filepath: str):
        """
        Save the episode buffer to a JSON file.
        
        Args:
            filepath: Path to the JSON file where the buffer will be saved.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        buffer_dict = self.to_dict()
        # Also save buffer metadata
        buffer_dict["buffer_size"] = self.buffer_size
        buffer_dict["num_episodes"] = len(self.buffer)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(buffer_dict, f, indent=2, ensure_ascii=False)
    
    # Load episode buffer from JSON file.
    @classmethod
    def load_from_json(cls, filepath: str, buffer_size: Optional[int] = None):
        """
        Load an episode buffer from a JSON file.
        
        Args:
            filepath: Path to the JSON file to load from.
            buffer_size: Optional buffer size. If None, uses the size from the file or defaults to 1000.
        
        Returns:
            Episode_Buffer instance loaded from the file.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Episode buffer file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            buffer_dict = json.load(f)
        
        # Get buffer size (prefer provided, then from file, then default)
        size = buffer_size
        if size is None:
            size = buffer_dict.get("buffer_size", 1000)
        
        # Create buffer instance
        buffer = cls(buffer_size=size)
        
        # Load episodes
        episodes_data = buffer_dict.get("episodes", [])
        buffer.buffer = [Episode.from_dict(ep_dict) for ep_dict in episodes_data]
        
        return buffer


# Store the tool/strategy extracted from the unlabeled experiences.
# Please note that the tool/strategy should have a limited length and a profile for retrieval.
class Tool_Buffer:
    def __init__(self, buffer_size: int):
        self.buffer = []
        self.buffer_size = buffer_size

    # Add tool(s) to the buffer.
    # Enforces buffer size limit using FIFO (First In First Out) policy.
    # Supports both single tool and list of tools (strategy-level).
    def add_tool(self, tool):
        # Handle both single tool and list of tools
        if isinstance(tool, list):
            # Add multiple tools at once
            self.buffer.extend(tool)
        elif isinstance(tool, SubTask_Experience):
            # Add SubTask_Experience as a tool
            self.buffer.append(tool)
        else:
            # Add single tool (string or object)
            self.buffer.append(tool)
        
        # Remove oldest tools if buffer exceeds maximum size
        if len(self.buffer) > self.buffer_size:
            overflow = len(self.buffer) - self.buffer_size
            self.buffer = self.buffer[overflow:]

    def add_tools(self, tools: List):
        """Add multiple tools at once (convenience method)."""
        self.add_tool(tools)

    def sample_tool(self, batch_size: int):
        """Sample random tools from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def get_tool_summary(self, query: str):
        return [tool.summary for tool in self.buffer
                if tool.summary and query in tool.summary]
    
    def __len__(self):
        """Return the number of tools in the buffer."""
        return len(self.buffer)

    # Convert the tool buffer to a dictionary.
    def to_dict(self):
        return {
            "tools": [tool.to_dict() for tool in self.buffer],
        }
    
    # Convert the dictionary to a tool buffer, while the content is the sub-task experience.
    def from_dict(self, d: dict):
        self.buffer = [SubTask_Experience.from_dict(tool) for tool in d["tools"]]
        return self
