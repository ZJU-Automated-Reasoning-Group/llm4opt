"""vLLM implementation for llm4opt."""

import os
from typing import Dict, List, Optional, Union, Any

from llm4opt.llm.base import LLM


class VLLM(LLM):
    """vLLM implementation.
    
    This class implements the LLM interface using vLLM for efficient inference.
    vLLM is a high-throughput and memory-efficient inference engine for LLMs.
    """
    
    def __init__(self, 
                model_name: str,
                tensor_parallel_size: int = 1,
                gpu_memory_utilization: float = 0.9,
                max_model_len: Optional[int] = None,
                quantization: Optional[str] = None,
                **kwargs):
        """Initialize the vLLM model.
        
        Args:
            model_name: The name or path of the model to use.
            tensor_parallel_size: Number of GPUs to use for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use for the model.
            max_model_len: Maximum sequence length to process.
            quantization: Quantization method to use (e.g., "int8", "int4").
            **kwargs: Additional parameters to pass to the vLLM LLM class.
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.kwargs = kwargs
        
        # Import vLLM here to avoid dependency issues if not installed
        try:
            from vllm import LLM as VLLMModel
            from vllm import SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Please install it with "
                "`pip install vllm` to use the VLLM class."
            )
            
        self.SamplingParams = SamplingParams
        
        # Initialize the vLLM model
        self.model = VLLMModel(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            quantization=quantization,
            **kwargs
        )
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: Optional[int] = None,
                stop: Optional[Union[str, List[str]]] = None,
                **kwargs) -> str:
        """Generate text from vLLM based on the prompt.
        
        Args:
            prompt: The prompt to generate text from.
            temperature: Controls randomness. Higher values (e.g., 0.8) make output more random,
                         lower values (e.g., 0.2) make it more deterministic.
            max_tokens: Maximum number of tokens to generate.
            stop: Sequences where the model will stop generating further tokens.
            **kwargs: Additional parameters to pass to the vLLM sampling parameters.
            
        Returns:
            The generated text as a string.
        """
        # Prepare sampling parameters
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else 1024,
            stop=stop,
            **kwargs
        )
        
        # Generate text
        outputs = self.model.generate(prompt, sampling_params)
        
        # Extract and return the generated text
        return outputs[0].outputs[0].text.strip()
    
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """Get the embedding for the given text.
        
        Note: vLLM doesn't natively support embeddings. This method raises a NotImplementedError.
        Consider using a dedicated embedding model or library like SentenceTransformers.
        
        Args:
            text: The text to get the embedding for.
            **kwargs: Additional parameters.
            
        Returns:
            The embedding as a list of floats.
            
        Raises:
            NotImplementedError: vLLM doesn't natively support embeddings.
        """
        raise NotImplementedError(
            "vLLM doesn't natively support embeddings. "
            "Consider using a dedicated embedding model or library like SentenceTransformers."
        )
