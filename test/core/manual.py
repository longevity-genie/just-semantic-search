from just_semantic_search.core.article_semantic_splitter import ArticleSemanticSplitter
import typer
from sentence_transformers import SentenceTransformer
from pprint import pprint
from just_semantic_search.core.embeddings import *
from just_semantic_search.core.utils import *
from just_semantic_search.core.text_splitter import TextSplitter
from pathlib import Path
import shutil
app = typer.Typer()


from typing import Union


def load_auto(model_name_or_path: str):
    import torch
from transformers import AutoTokenizer, AutoModel


@app.command()
def main():
    """
    Main function.
    """
    # Get the current file's directory and construct path to data file
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent.parent  # Go up 3 levels from test/core to project root
    data_file = project_dir / "data" / "tacutopapers_test_rsids_10k" / "8.txt"
    
    model: SentenceTransformer = load_gte_mlm_en()
    print("model: ", model)
    print("dimensions: ", model.get_sentence_embedding_dimension())
    print("max_seq_length: ", model.max_seq_length)
    tmp = current_dir / "tmp" 
    # Clean up existing tmp directory if it exists
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(exist_ok=True)
    
    splitter = ArticleSemanticSplitter(model)
    abstract = """
    Multiple studies characterizing the human ageing phenotype have been conducted for decades. However, there is no centralized resource in which data on multiple age-related changes are collated. Currently, researchers must consult several sources, including primary publications, in order to obtain age-related data at various levels. To address this and facilitate integrative, system-level studies of ageing we developed the Digital Ageing Atlas (DAA). The DAA is a one-stop collection of human age-related data covering different biological levels (molecular, cellular, physiological, psychological and pathological) that is freely available online (http://ageing-map.org/). Each of the >3000 age-related changes is associated with a specific tissue and has its own page displaying a variety of information, including at least one reference. Age-related changes can also be linked to each other in hierarchical trees to represent different types of relationships. In addition, we developed an intuitive and user-friendly interface that allows searching, browsing and retrieving information in an integrated and interactive fashion. Overall, the DAA offers a new approach to systemizing ageing resources, providing a manually-curated and readily accessible source of age-related changes.
    """
    title = "The Digital Ageing Atlas: integrating the diversity of age-related changes into a unified resource"
    source = "https://doi.org/10.1093/nar/gku843"
    documents = splitter.split_file(data_file, embed=True, abstract=abstract, title=title, source=source)
    for i, document in enumerate(documents):
        #print(f"=============SHAPE:======={document.vectors.shape}=========================")
        #pprint(document.vectors)
        print(document.content)
        print("===============================================")
        file_path = current_dir / "tmp" / (data_file.stem+ f"{i}.yaml")
        document.save_to_yaml(file_path)
    #typer.echo(f"Hello {name}!")

if __name__ == "__main__":
    app()
