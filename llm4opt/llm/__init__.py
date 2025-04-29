"""LLM module for llm4opt."""

from llm4opt.llm.base import LLM
from llm4opt.llm.gpt import GPT
from llm4opt.llm.vllm import VLLM
from llm4opt.llm.deepseek import DeepSeek
from llm4opt.llm.zhipu import ZhipuLLM
from llm4opt.llm.openrouter import OpenRouter

__all__ = ["LLM", "GPT", "VLLM", "DeepSeek", "ZhipuLLM", "OpenRouter"]
