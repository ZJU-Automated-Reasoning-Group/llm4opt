"""DeepSeek implementation for llm4opt."""

import os
from typing import Dict, List, Optional, Union, Any

import requests

from llm4opt.llm.base import LLM


class DeepSeek(LLM):
    """DeepSeek implementation.
    
    This class implements the LLM interface using DeepSeek's API.
    """
    
    def __init__(self, 
                model_name: str = "deepseek-coder",
                api_key: Optional[str] = None,
                api_base: Optional[str] = None,
                **kwargs):
        """Initialize the DeepSeek model.
        
        Args:
            model_name: The name of the DeepSeek model to use (default: "deepseek-coder").
            api_key: DeepSeek API key. If not provided, it will be read from the
                     DEEPSEEK_API_KEY environment variable.
            api_base: DeepSeek API base URL. If not provided, it will use the default.
            **kwargs: Additional parameters to pass to the API.
        """
        self.model_name = model_name
        
        # Set API key from parameter or environment variable
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is required. Either pass it as api_key parameter "
                "or set the DEEPSEEK_API_KEY environment variable."
            )
        
        # Set API base URL
        self.api_base = api_base or "https://api.deepseek.com/v1"
        
        # Store additional kwargs
        self.kwargs = kwargs
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: Optional[int] = None,
                stop: Optional[Union[str, List[str]]] = None,
                **kwargs) -> str:
        """Generate text from DeepSeek based on the prompt.
        
        Args:
            prompt: The prompt to generate text from.
            temperature: Controls randomness. Higher values (e.g., 0.8) make output more random,
                         lower values (e.g., 0.2) make it more deterministic.
            max_tokens: Maximum number of tokens to generate.
            stop: Sequences where the API will stop generating further tokens.
            **kwargs: Additional parameters to pass to the DeepSeek API.
            
        Returns:
            The generated text as a string.
        """
        # Prepare the API endpoint
        endpoint = f"{self.api_base}/chat/completions"
        
        # Prepare the request headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        # Add optional parameters if provided
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
            
        # Add any additional kwargs
        payload.update(kwargs)
        
        # Make the API call
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the response
        response_json = response.json()
        
        # Extract and return the generated text
        return response_json["choices"][0]["message"]["content"].strip()
    
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """Get the embedding for the given text using DeepSeek's embedding models.
        
        Args:
            text: The text to get the embedding for.
            **kwargs: Additional parameters to pass to the DeepSeek API.
            
        Returns:
            The embedding as a list of floats.
        """
        # Prepare the API endpoint
        endpoint = f"{self.api_base}/embeddings"
        
        # Prepare the request headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Default to the embedding model if not specified
        embedding_model = kwargs.pop("embedding_model", "deepseek-embedding")
        
        # Prepare the request payload
        payload = {
            "model": embedding_model,
            "input": text,
        }
        
        # Add any additional kwargs
        payload.update(kwargs)
        
        # Make the API call
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the response
        response_json = response.json()
        
        # Extract and return the embedding
        return response_json["data"][0]["embedding"]
    