import typer
from sentence_transformers import SentenceTransformer
from pprint import pprint
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from typing import Tuple, Union

def load_auto_model_tokenizer(model_name_or_path: str, trust_remote_code: bool = True) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    return model, tokenizer

def load_sentence_transformer_model(model_name_or_path: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
    return model

DEFAULT_EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"


#def load_BioBERT() -> SentenceTransformer:
#    return load_sentence_transformer_model("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

#def load_bio_lord() -> SentenceTransformer:
#    return load_sentence_transformer_model("FremyCompany/BioLORD-2023")

def load_gte_multilingual() -> SentenceTransformer:
    return load_sentence_transformer_model("Alibaba-NLP/gte-multilingual-base")

def load_gte_multilingual_mlm() -> SentenceTransformer:
    return load_sentence_transformer_model("Alibaba-NLP/gte-multilingual-mlm-base")

def load_gte_mlm_en() -> SentenceTransformer:
    return load_sentence_transformer_model("Alibaba-NLP/gte-en-mlm-large")

def load_gte_large() -> SentenceTransformer:
    return load_sentence_transformer_model("Alibaba-NLP/gte-large-en-v1.5")


def load_specter() -> SentenceTransformer:
    return load_sentence_transformer_model("sentence-transformers/allenai-specter")

def load_bioembeddings() -> SentenceTransformer:
    return load_sentence_transformer_model("pavanmantha/bge-base-en-bioembed")

#def load_biomedlm():
#    return load_sentence_transformer_model("stanford-crfm/BioMedLM")

def load_medcpt_query_model_with_tokenizer() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    return load_auto_model_tokenizer("ncbi/MedCPT-Query-Encoder")

def load_medcpt_article_model_with_tokenizer() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    return load_auto_model_tokenizer("MedCPT-Article-Encoder")


def load_jinaai_embeddings_v3() -> SentenceTransformer:
    return load_sentence_transformer_model("jinaai/jina-embeddings-v3")

