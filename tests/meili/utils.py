from just_semantic_search.article_semantic_splitter import ArticleSemanticSplitter
from just_semantic_search.article_splitter import ArticleSplitter
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
from pathlib import Path
import time
from eliot import start_action, log_message

from just_semantic_search.meili.rag import *
from typing import Union
from just_semantic_search.meili.rag import *
import time
from pathlib import Path
from just_semantic_search.article_semantic_splitter import ArticleSemanticSplitter
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
from pathlib import Path
import time
#from just_semantic_search.utils import RenderingFileDestination
from just_semantic_search.article_semantic_splitter import ArticleSemanticSplitter
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
from pathlib import Path
import time
#from just_semantic_search.utils import RenderingFileDestination


import typer
import os
from just_semantic_search.meili.rag import *
from just_semantic_search.meili.rag import *
import time
from pathlib import Path
import subprocess
import requests
from eliot import log_message

from eliot._output import *
from eliot import start_action, start_task

import typer
import os
from just_semantic_search.meili.rag import *
from just_semantic_search.meili.rag import *
import time
from pathlib import Path
import subprocess
import requests
from eliot import log_message
import sys

from eliot._output import *
from eliot import start_action, start_task

from typing import Union

def split_and_print_documents(splitter: Union[ArticleSplitter, ArticleSemanticSplitter], 
                            data_file: Path, 
                            tmp_dir: Path, 
                            abstract: str, 
                            title: str, 
                            source: str) -> float:
    """
    Splits the given data file into documents, prints their content, and saves them to YAML files.
    Returns the time taken to split (excluding file I/O operations).
    """
    with start_action(action_type="split_documents",
                     data_file=str(data_file),
                     splitter_type=splitter.__class__.__name__):
        
        # Measure only the splitting time
        start_time = time.time()
        documents = splitter.split_file(data_file, embed=True, abstract=abstract, title=title, source=source)
        end_time = time.time()
        split_time = end_time - start_time
        
        log_message(message_type="split_complete",
                   split_time=split_time,
                   document_count=len(documents))
        
        # File I/O operations after timing measurement
        for i, document in enumerate(documents):
            with start_action(action_type="save_document",
                            fragment_number=i,
                            token_count=document.token_count,
                            document_title=document.title):
                file_path = tmp_dir / f"{data_file.stem}_{i}.yaml"
                document.save_to_yaml(file_path)
        
        return split_time
    
