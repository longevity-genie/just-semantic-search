import json
import pprint

from dotenv import load_dotenv

from just_agents import llm_options
from just_agents.base_agent import BaseAgent
from just_semantic_search.embeddings import EmbeddingModel
from just_semantic_search.meili.tools import all_indexes, search_documents
import os
load_dotenv(override=True)



if __name__ == "__main__":
    
    ensure_meili_is_running(meili_service_dir)
    
    # Example prompt asking about weather in multiple cities
    prompt = "What is a digital ageing atlas?"
    agent = BaseAgent(  # type: ignore
        llm_options=llm_options.LLAMA3_3,
        tools=[search_documents, all_indexes],
        system_prompt="You are a helpful assistant that can search for documents in a MeiliSearch database. You can only use index robi-tacutu as we can use only robi tacutu papers."
    )

    
   
    # Add a callback to print messages using pprint
    # This will show the internal conversation/function calls
    agent.memory.add_on_message(lambda m: pprint.pprint(m))
    
    # query with memory callback enabled
    result = agent.query(prompt)
    print("RESULT+++++++++++++++++++++++++++++++++++++++++++++++")
    print(result)
   