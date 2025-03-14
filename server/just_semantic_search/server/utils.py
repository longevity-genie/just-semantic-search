import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List
import time
from just_agents import llm_options
from just_agents.llm_options import LLAMA3_3
from just_agents.web.chat_ui_agent import ChatUIAgent
from just_semantic_search.meili.tools import search_documents, all_indexes
from just_semantic_search.meili.utils.services import ensure_meili_is_running

def load_environment_files(env_path=None):
    """
    Load environment variables from various possible .env files
    
    Args:
        env_path: Optional specific path to an env file to try first
        
    Returns:
        bool: Whether any environment file was successfully loaded
    """
    # Define possible env file names to check
    env_file_names = [".env.keys", ".env.local", ".env"]
    env_loaded = False
    
    # Try to load from the configured path first if provided
    if env_path:
        env_path = Path(env_path).resolve().absolute()
        if env_path.exists():
            load_dotenv(env_path, override=True)
            env_loaded = True
    
    if not env_loaded:
        # Get the current directory and check for env files
        current_dir = Path(__file__).parent.resolve().absolute()
        
        # Check directories in order: current directory, parent, parent's parent
        directories_to_check = [
            current_dir,
            current_dir.parent,
            current_dir.parent.parent
        ]
        
        for directory in directories_to_check:
            if env_loaded:
                break
                
            for env_name in env_file_names:
                potential_env_file = directory / env_name
                if potential_env_file.exists():
                    load_dotenv(potential_env_file, override=True)
                    env_loaded = True
                    break
    
    # If no specific env file was found, fall back to default behavior
    if not env_loaded:
        load_dotenv(override=True)
        
    return env_loaded

def get_project_directories():
    """
    Get standard project directories based on environment variables
    
    Returns:
        dict: Dictionary containing project directory paths
    """
    current_dir = Path(__file__).parent
    project_dir = Path(os.getenv("APP_DIR", str(current_dir.parent.parent.parent))).absolute()
    data_dir = project_dir / os.getenv("DATA_DIR", "data")
    logs_dir = project_dir / os.getenv("LOG_DIR", "logs")
    meili_service_dir = project_dir / "meili"
    
    # Create directories if they don't exist
    for directory in [data_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return {
        "current_dir": current_dir,
        "project_dir": project_dir,
        "data_dir": data_dir,
        "logs_dir": logs_dir,
        "meili_service_dir": meili_service_dir
    }

def default_rag_agent():
    """
    Create a default RAG agent with standard configuration
    
    Returns:
        ChatUIAgent: Configured RAG agent
    """
    call_indexes = "YOU DO NOT search documents until you will retrive all the indexes in the database. When you search you are only alllowed to select from the indexes that you retrived, do not invent indexes!"
    
    return ChatUIAgent(
        llm_options=LLAMA3_3, #llm_options.GEMINI_2_FLASH,
        tools=[search_documents, all_indexes],
        system_prompt=f"""
        You are a helpful assistant that can search for documents in a MeiliSearch database. 
        {call_indexes}
        You can only search indexes that you got from all_indexes tool, do not invent indexes that do not exist.
        You MUST ALWAYS provide sources for all the documents. Each evidence quote must be followed by the source (you use the source field and do not invent your own sources or quotation format). 
        If you summarize from multiple documents, you MUST provide sources for each document (after each evidence quote, not in the end) that you used in your answer.
        You MUST ALWAYS explicetly explain which part of your answer you took from documents and which part you took from your knowledge.
        YOU NEVER CALL THE TOOL WITH THE SAME PARAMETERS MULTIPLE TIMES.
        The search document tool uses semantic search.
        """
    )

def default_annotation_agent():
    """
    Create a default annotation agent with standard configuration
    
    Returns:
        ChatUIAgent: Configured annotation agent
    """
    return ChatUIAgent(
        llm_options=LLAMA3_3, #llm_options.GEMINI_2_FLASH,
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
        File filename you give a filename of the file in the folder together with the extension."""
    )

def setup_meili(meili_dir: Optional[Path] = None, ensure: bool = True):
    """
    Ensure MeiliSearch is running and return available indexes
    
    Args:
        meili_dir: Directory containing MeiliSearch service
        ensure: Whether to ensure MeiliSearch is running
        
    Returns:
        list: Available indexes
    """
    if meili_dir is None:
        dirs = get_project_directories()
        meili_dir = dirs["meili_service_dir"]
        
    if ensure:
        ensure_meili_is_running(meili_dir)
    
    # Get available indexes
    return all_indexes(non_empty=True)

def time_function(func, *args, **kwargs):
    """
    Time the execution of a function
    
    Args:
        func: Function to time
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        tuple: (result, execution_time_in_seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time 