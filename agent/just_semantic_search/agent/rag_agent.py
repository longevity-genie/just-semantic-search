import os

import json
from pathlib import Path
import pprint
import warnings

# Suppress all warnings from torch, transformers and flash-attn
warnings.filterwarnings('ignore', message='.*flash_attn.*')

from dotenv import load_dotenv

from just_agents import llm_options
from just_agents.base_agent import BaseAgent
from just_semantic_search.embeddings import EmbeddingModel
from just_semantic_search.meili.tools import all_indexes, search_documents, search_documents_debug
import os

from just_semantic_search.meili.utils.services import ensure_meili_is_running
load_dotenv(override=True)

current_dir = Path(__file__).parent
project_dir = current_dir.parent.parent.parent  # Go up 2 levels from test/meili to project root
data_dir = project_dir / "data"
logs = project_dir / "logs"
tacutopapers_dir = data_dir / "tacutopapers_test_rsids_10k"
meili_service_dir = project_dir / "services" / "meili"



if __name__ == "__main__":
    load_dotenv(override=True)
    
    # Add warning filter before creating the agent
    warnings.filterwarnings("ignore", message="flash_attn is not installed")
    
    ensure_meili_is_running(meili_service_dir)
    
    
    agent = BaseAgent(  # type: ignore
        llm_options=llm_options.OPENAI_GPT4o,
        tools=[search_documents, all_indexes],
        system_prompt="""
        You are a helpful assistant that can search for documents in a MeiliSearch database. You can only search in glucosedao index. It is especially usefull for CGM-related questions.
        You MUST ALWAYS provide sources for all the documents. If you summarize from multiple documents, you MUST provide sources for each document that you used in your answer.
        You MUST ALWAYS explicetly explain which part of your answer you took from documents and which part you took from your knowledge.
        YOU NEVER CALL THE TOOL WITH THE SAME PARAMETERS MULTIPLE TIMES.
        The search document function uses semantic search.
        """
    )

    prompt = "Which machine learning models are used for CGM?"
   
    # Add a callback to print messages using pprint
    # This will show the internal conversation/function calls
    agent.memory.add_on_message(lambda m: pprint.pprint(m))
    
    # query with memory callback enabled
    result = agent.query(prompt)
    #print("RESULT+++++++++++++++++++++++++++++++++++++++++++++++")
    #print(result)
   