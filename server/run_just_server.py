#!/usr/bin/env python
"""
Run Just Semantic Search Server

This script provides a simple way to run the Just Semantic Search RAG server
with support for multiple workers. It uses Typer for command-line handling
and properly implements the module:app pattern for Uvicorn when running with
multiple workers.

Examples:
  # Run with default settings (single worker)
  python run_just_server.py
  
  # Run with 4 workers on port 8080
  python run_just_server.py --port 8080 --workers 4
  
  # Run in debug mode
  python run_just_server.py --debug
"""

import sys
import os

# Add the current directory to the Python path to make imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the run_server_command directly for cleaner execution
from just_semantic_search.server.run_rag_server import run_server_with_cli

if __name__ == "__main__":
    # Run the server with command line arguments
    run_server_with_cli() 