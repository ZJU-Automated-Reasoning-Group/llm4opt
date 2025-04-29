"""DeepSeek implementation."""

import os
from typing import List, Optional, Union
import requests

from llm4opt.llm.base import LLM


class DeepSeek(LLM):
    def __init__(self, 
                model_name: str = "deepseek-coder",
                api_key: Optional[str] = None,
                api_base: Optional[str] = None,
                **kwargs):
        self.model_name = model_name
        
        # Set API key from parameter or environment variable
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required via api_key parameter or DEEPSEEK_API_KEY environment variable")
        
        # Set API base URL
        self.api_base = api_base or "https://api.deepseek.com/v1"
        self.kwargs = kwargs
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: Optional[int] = None,
                stop: Optional[Union[str, List[str]]] = None,
                **kwargs) -> str:
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
    