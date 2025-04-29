"""Base class for LLM implementations in llm4opt."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any


class LLM(ABC):
    """Base class for LLM implementations.
    
    This abstract class defines the interface that all LLM implementations
    should follow in the llm4opt package.
    """
    
    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        """Initialize the LLM.
        
        Args:
            model_name: The name of the model to use.
            **kwargs: Additional model-specific parameters.
        """
        pass
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: Optional[int] = None,
                stop: Optional[Union[str, List[str]]] = None,
                **kwargs) -> str:
        """Generate text from the LLM based on the prompt.
        
        Args:
            prompt: The prompt to generate text from.
            temperature: Controls randomness. Higher values (e.g., 0.8) make output more random,
                         lower values (e.g., 0.2) make it more deterministic.
            max_tokens: Maximum number of tokens to generate.
            stop: Sequences where the API will stop generating further tokens.
            **kwargs: Additional model-specific parameters.
            
        Returns:
            The generated text as a string.
        """
        pass
    
    @abstractmethod
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """Get the embedding for the given text.
        
        Args:
            text: The text to get the embedding for.
            **kwargs: Additional model-specific parameters.
            
        Returns:
            The embedding as a list of floats.
        """
        pass
