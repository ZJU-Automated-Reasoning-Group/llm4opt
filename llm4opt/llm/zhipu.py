"""Zhipu AI API"""


from .base import LLM


class Zhipu(LLM):
    def __init__(self, api_key: str, model_name: str = "deepseek-coder"):
        self.api_key = api_key
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the LLM based on the prompt."""
        pass

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """Get the embedding of the text."""
        pass