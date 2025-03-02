import os

from pathlib import Path
import pprint
import warnings
import typer
from typing import Optional
from just_agents.web.chat_ui_agent import ChatUIAgent

# Suppress all warnings from torch, transformers and flash-attn
warnings.filterwarnings('ignore', message='.*flash_attn.*')

from dotenv import load_dotenv

from just_agents import llm_options
from just_agents.llm_options import LLAMA3_3, GEMINI_2_FLASH, GEMINI_2_FLASH_EXP
from just_agents.web.web_agent import WebAgent
from just_semantic_search.meili.tools import all_indexes, search_documents
import os
from just_agents import tools
from just_agents.tools.semantic_search import semantic_search, list_search_indexes

from just_semantic_search.meili.utils.services import ensure_meili_is_running
load_dotenv(override=True)


current_dir = Path(__file__).parent
project_dir = Path(os.getenv("APP_DIR", str(current_dir.parent.parent.parent))).absolute()   # Go up 2 levels from test/meili to project root
data_dir = project_dir / os.getenv("DATA_DIR", "data")
logs = project_dir / os.getenv("LOG_DIR", "logs")
tacutopapers_dir = data_dir / "tacutopapers_test_rsids_10k"
meili_service_dir = project_dir / "meili"


call_indexes = "YOU DO NOT search documents until you will retrive all the indexes in the database. When you search you are only alllowed to select from the indexes that you retrived, do not invent indexes!"

def default_rag_agent(): 
    return ChatUIAgent(
            llm_options=llm_options.GEMINI_2_FLASH,
            tools=[search_documents, all_indexes],
            system_prompt=f"""
            You are a helpful assistant that can search for documents in a MeiliSearch database. 
            f{call_indexes}
            You can only search indexes that you got from all_indexes function, do not invent indexes that do not exist.
            You MUST ALWAYS provide sources for all the documents. Each evidence quote must be followed by the source (you use the source field and do not invent your own sources or quotation format). 
            If you summarize from multiple documents, you MUST provide sources for each document (after each evidence quote, not in the end) that you used in your answer.
            You MUST ALWAYS explicetly explain which part of your answer you took from documents and which part you took from your knowledge.
            YOU NEVER CALL THE TOOL WITH THE SAME PARAMETERS MULTIPLE TIMES.
            The search document function uses semantic search.
            """
        )

def default_annotation_agent():
    return ChatUIAgent(
        llm_options=llm_options.GEMINI_2_FLASH,
        tools=[],   
        system_prompt="""You are a paper annotator. You extract the abstract, authors and titles of the papers.
            Abstract and authors must be exactly he way they are in the paper, do not edit them.
            You provide your output as json object of the following JSON format:
            {
                "abstract": "...",
                "authors": ["...", "..."],
                "title": "...",
                "source": "...",
            }
            Make sure to provide the output in the correct format, do not add any other text or comments, do not add ```json or other surrounding.
            For string either use one line or use proper escape characters (\n) for line breaks
            Make sure to provide the output in the correct format, do not add any other text or comments.
            For source you either give DOI, pubmed or filename (if doi or pubmed is not available).
            File filename you give a filename of the file in the folder together with the extension.""",
            
)

app = typer.Typer()

@app.command()
def query_agent(
    prompt: str = typer.Argument(default="Which machine learning models are used for CGM?", help="The question to ask the agent"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output with pprint"),
    model: str = typer.Option("gemini/gemini-2.0-flash", "--model", "-m", help="LLM model to use"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature for LLM generation"),
    ensure_meili: bool = typer.Option(True, "--ensure-meili/--no-ensure-meili", help="Ensure MeiliSearch is running"),
    meili_dir: Optional[Path] = typer.Option(
        meili_service_dir, 
        "--meili-dir",
        help="Directory containing MeiliSearch service",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
):
    """
    Query the RAG agent with a prompt and get a response.
    """
    load_dotenv(override=True)
    
    if ensure_meili:
        ensure_meili_is_running(meili_dir)
    
    # Configure the agent
    llm_config = {
        "model": model,
        "temperature": temperature
    }
    indexes = all_indexes(non_empty=True)
    rag_agent = default_rag_agent()
    
    agent = WebAgent(
        llm_options=llm_config,
        tools=[search_documents, all_indexes],
        system_prompt=rag_agent.system_prompt.replace(call_indexes, f"You can only search indexes: {indexes}. NEVER put index parameter in the search function which is not in this list.")
    )
    
    
    if debug:
        agent.memory.add_on_message(lambda role, message: pprint.pprint(message))
    
    result = agent.query(prompt)
    print(result)

if __name__ == "__main__":

    #agent = WebAgent(
    #    llm_options=LLAMA3_3,
    #    system_prompt="""
    #    You MUST ALWAYS provide sources for all the documents. Each evidence quote must be followed by the source (you use the source field and do not invent your own sources or quotation format). 
    #    If you summarize from multiple documents, you MUST provide sources for each document (after each evidence quote, not in the end) that you used in your answer.
    #    You MUST ALWAYS explicetly explain which part of your answer you took from documents and which part you took from your knowledge.
    #    YOU NEVER CALL THE TOOL WITH THE SAME PARAMETERS MULTIPLE TIMES.
    #    You only use semantic_search when you know which indexes are available. Please, call list_search_indexes tool first and then use semantic_search.
    #    """,
    #    tools=[semantic_search, list_search_indexes],
    #)
    #print("=====================================")
    #result = agent.query("Which machine learning models are used for CGM?")
    #print(result)
    #agent.memory.pretty_print_all_messages()

    print("Starting RAG agent")
    app()

   