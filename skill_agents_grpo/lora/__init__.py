"""
Multi-LoRA infrastructure for the skill-bank agent.

One shared Qwen-3-8B backbone + four function-specific LoRA adapters
(boundary, segment, contract, retrieval).

Quick start::

    from skill_agents_grpo.lora import MultiLoraSkillBankLLM, MultiLoraConfig, SkillFunction

    cfg = MultiLoraConfig(adapter_paths={"boundary": "path/to/adapter"})
    llm = MultiLoraSkillBankLLM(cfg)
    text = llm.generate(SkillFunction.BOUNDARY, "Your prompt here")
"""

from skill_agents_grpo.lora.skill_function import SkillFunction
from skill_agents_grpo.lora.config import MultiLoraConfig, LoraTrainingConfig
from skill_agents_grpo.lora.model import MultiLoraSkillBankLLM

__all__ = [
    "SkillFunction",
    "MultiLoraConfig",
    "LoraTrainingConfig",
    "MultiLoraSkillBankLLM",
]
