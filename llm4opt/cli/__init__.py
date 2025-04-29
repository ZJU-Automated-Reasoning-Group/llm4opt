"""Command-line interface tools for the llm4opt package.

This module provides CLI tools for working with the llm4opt package, including:
- gen_hints: Generate compiler hints for C/C++ code
"""

from llm4opt.cli.gen_hints import main as gen_hints_main

__all__ = ['gen_hints_main'] 