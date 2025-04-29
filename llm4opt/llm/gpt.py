"""OpenAI GPT implementation for llm4opt."""

import os
from typing import Dict, List, Optional, Union, Any

import openai

from llm4opt.llm.base import LLM


class GPT(LLM):
    """OpenAI GPT implementation.
    
    This class implements the LLM interface using OpenAI's GPT models.
    """
    
    def __init__(self, 
                model_name: str = "gpt-4-turbo", 
                api_key: Optional[str] = None,
                organization: Optional[str] = None,
                **kwargs):
        """Initialize the GPT model.
        
        Args:
            model_name: The name of the OpenAI model to use (default: "gpt-4-turbo").
            api_key: OpenAI API key. If not provided, it will be read from the
                     OPENAI_API_KEY environment variable.
            organization: OpenAI organization ID. If not provided, it will be read from the
                          OPENAI_ORGANIZATION environment variable if available.
            **kwargs: Additional parameters to pass to the OpenAI client.
        """
        self.model_name = model_name
        
        # Set API key from parameter or environment variable
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Either pass it as api_key parameter "
                "or set the OPENAI_API_KEY environment variable."
            )
        
        # Set organization from parameter or environment variable
        self.organization = organization or os.environ.get("OPENAI_ORGANIZATION")
        
        # Initialize OpenAI client
        client_kwargs = {}
        if self.organization:
            client_kwargs["organization"] = self.organization
            
        self.client = openai.OpenAI(api_key=self.api_key, **client_kwargs)
        
        # Store additional kwargs
        self.kwargs = kwargs
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: Optional[int] = None,
                stop: Optional[Union[str, List[str]]] = None,
                **kwargs) -> str:
        """Generate text from GPT based on the prompt.
        
        Args:
            prompt: The prompt to generate text from.
            temperature: Controls randomness. Higher values (e.g., 0.8) make output more random,
                         lower values (e.g., 0.2) make it more deterministic.
            max_tokens: Maximum number of tokens to generate.
            stop: Sequences where the API will stop generating further tokens.
            **kwargs: Additional parameters to pass to the OpenAI API.
            
        Returns:
            The generated text as a string.
        """
        # Prepare parameters for the API call
        params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        # Add optional parameters if provided
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if stop is not None:
            params["stop"] = stop
            
        # Add any additional kwargs
        params.update(kwargs)
        
        # Make the API call
        response = self.client.chat.completions.create(**params)
        
        # Extract and return the generated text
        return response.choices[0].message.content.strip()
    
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """Get the embedding for the given text using OpenAI's embedding models.
        
        Args:
            text: The text to get the embedding for.
            **kwargs: Additional parameters to pass to the OpenAI API.
            
        Returns:
            The embedding as a list of floats.
        """
        # Default to text-embedding-3-small if not specified
        embedding_model = kwargs.pop("embedding_model", "text-embedding-3-small")
        
        # Make the API call
        response = self.client.embeddings.create(
            model=embedding_model,
            input=text,
            **kwargs
        )
        
        # Extract and return the embedding
        return response.data[0].embedding
