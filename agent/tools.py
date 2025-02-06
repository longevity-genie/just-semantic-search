import os
from typing import Optional
from just_semantic_search.meili.tools import search_documents
from just_semantic_search.agent.rag_agent import BaseAgent
from just_semantic_search.agent import llm_options

# TODO work on this advanced agent-based summary tool
# as it is not implemented yet

DEFAULT_SUMMARY_PROMPT = """
You are a search reranker and summarizer. You are given a query and a search result.
You must evaluate the relevance of search results from 0 to 100 percent. Keep source, authors and titles.
Extract from text exact pieces relevant to the query.
The output must be in JSON format:
{
    "summary": "...",
    "authors": ["...", "..."],
    "title": ["...", "..."],
    "relevance": [0, 100]
    "source": ["..."],
    
}
"""

def make_summary_tool(summary_prompt: str = DEFAULT_SUMMARY_PROMPT, llm_options: llm_options.LLMOptions = llm_options.LLAMA3_3):
    """
    Make a tool that summarizes the search results.
    """
     agent = BaseAgent(  # type: ignore
        llm_options=llm_options,
        system_prompt=summary_prompt,
        tools=[search_documents]
    )
    # should geerate a tool code that uses agent inside
    pass
