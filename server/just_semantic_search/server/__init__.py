"""
Just Semantic Search Server
===========================

This package contains server implementations for semantic search functionality,
including RAG (Retrieval Augmented Generation) capabilities.
"""

# Make key components available at the package level
from just_semantic_search.server.rag_server import RAGServer, RAGServerConfig
from just_semantic_search.server.run_rag_server import create_app, get_app 