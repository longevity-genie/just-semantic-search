from just_semantic_search.article_semantic_splitter import ArticleSemanticSplitter
from just_semantic_search.article_slitter import ArticleSplitter
from just_semantic_search.document import ArticleDocument
from sentence_transformers import SentenceTransformer
from pprint import pprint
from just_semantic_search.embeddings import *
from just_semantic_search.utils import *
from pathlib import Path
import shutil
import time


import typer
import os
from dotenv import load_dotenv
from just_semantic_search.meili.rag import *
from typing import List, Dict, Any, Union
from just_semantic_search.meili.rag import *
import time
from pathlib import Path
load_dotenv(override=True)
key = os.getenv("MEILI_MASTER_KEY", "fancy_master_key")

app = typer.Typer()


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
    # Measure only the splitting time
    start_time = time.time()
    documents = splitter.split_file(data_file, embed=True, abstract=abstract, title=title, source=source)
    end_time = time.time()
    split_time = end_time - start_time
    
    # File I/O operations after timing measurement
    for i, document in enumerate(documents):
        print(f"===FRAGMENT_{i}_tokens={document.token_count}===TITLE: {document.title}===========================")
        file_path = tmp_dir / f"{data_file.stem}_{i}.yaml"
        document.save_to_yaml(file_path)
    
    return split_time


@app.command()
def main():
    """
    Main function.
    """

    # Get the current file's directory and construct path to data file
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent  # Go up 3 levels from test/core to project root
    data_file = project_dir / "data" / "tacutopapers_test_rsids_10k" / "108.txt"

    model: SentenceTransformer = load_gte_large()
    print("model: ", model)
    print("dimensions: ", model.get_sentence_embedding_dimension())
    print("max_seq_length: ", model.max_seq_length)
    device = model.device
    print(f"Model is running on: {device}")
    tmp = current_dir / "tmp" 
    # Clean up existing tmp directory if it exists
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(exist_ok=True)

    abstract = """
    Multiple studies characterizing the human ageing phenotype have been conducted for decades. However, there is no centralized resource in which data on multiple age-related changes are collated. Currently, researchers must consult several sources, including primary publications, in order to obtain age-related data at various levels. To address this and facilitate integrative, system-level studies of ageing we developed the Digital Ageing Atlas (DAA). The DAA is a one-stop collection of human age-related data covering different biological levels (molecular, cellular, physiological, psychological and pathological) that is freely available online (http://ageing-map.org/). Each of the >3000 age-related changes is associated with a specific tissue and has its own page displaying a variety of information, including at least one reference. Age-related changes can also be linked to each other in hierarchical trees to represent different types of relationships. In addition, we developed an intuitive and user-friendly interface that allows searching, browsing and retrieving information in an integrated and interactive fashion. Overall, the DAA offers a new approach to systemizing ageing resources, providing a manually-curated and readily accessible source of age-related changes.
    """
    title = "The Digital Ageing Atlas: integrating the diversity of age-related changes into a unified resource"
    source = "https://doi.org/10.1093/nar/gku843"

    
    # First splitter
    
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(exist_ok=True)

      
    splitter = ArticleSemanticSplitter(model)

    documents = splitter.split_file(data_file, embed=True, abstract=abstract, title=title, source=source)

    #test_documents_index(documents, splitter.model_name, "test", "127.0.0.1", 7700, text_query="How are CAD-genes involved in aging?")
    host = "127.0.0.1"
    port = 7700
    model_name = splitter.model_name
    text_query = "Batman"
    #text_query = "How are CAD-genes involved in aging?"
    index_name = "test"

    config = MeiliConfig(host=host, port=port, api_key=key)
    rag = MeiliRAG("test", model_name, config, create_index_if_not_exists=True, recreate_index=True)
    
    # Ensure documents have vectors with the correct model name key
    for doc in documents:
        if model_name not in doc.vectors:
            print(f"Warning: Document missing vector for model {model_name}")
    
    count = rag.add_documents(documents)
    print(f"Added {count} documents to the index")
   

    rag.index.update_searchable_attributes(['title', 'abstract', 'text'])
    # Test search
    vector_query = model.encode("What are CAD-genes?")

    results = rag.search(None, vector=vector_query)
    print(f"\nSearch results for '{text_query}' in index '{index_name}':")
    for hit in results.hits:
       print("HIT:")
       pprint(hit)
    
    return results
    

if __name__ == "__main__":
    app()
