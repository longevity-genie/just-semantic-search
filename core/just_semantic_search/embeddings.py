from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from typing import Tuple, Union
from enum import Enum

def load_auto_model_tokenizer(model_name_or_path: str, trust_remote_code: bool = True) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    return model, tokenizer

def load_sentence_transformer_model(model_name_or_path: str, cache_folder: str = None, **model_kwargs) -> SentenceTransformer:
    model = SentenceTransformer(model_name_or_path, trust_remote_code=True, cache_folder=cache_folder, **model_kwargs)
    
    # Try to apply PyTorch 2.0+ compilation if available
    #try:
    #    import torch
    #    if hasattr(torch, 'compile'):
    #        # Instead of compiling `model.model`, compile the entire SentenceTransformer instance
    #        model = torch.compile(model)
    #        print(f"Applied torch.compile() optimization to {model_name_or_path}")
    #except Exception as e:
    #    print(f"Couldn't apply torch.compile: {e}")
        
    return model



class EmbeddingModel(Enum):
    GTE_LARGE = "Alibaba-NLP/gte-large-en-v1.5"
    GTE_MULTILINGUAL = "Alibaba-NLP/gte-multilingual-base"
    GTE_MULTILINGUAL_MLM = "Alibaba-NLP/gte-multilingual-mlm-base"
    GTE_MLM_EN = "Alibaba-NLP/gte-en-mlm-large"
    SPECTER = "sentence-transformers/allenai-specter"
    BIOEMBEDDINGS = "pavanmantha/bge-base-en-bioembed"
    MODERN_BERT_LARGE = "answerdotai/ModernBERT-large"
    JINA_EMBEDDINGS_V3 = "jinaai/jina-embeddings-v3"
    MEDCPT_QUERY = "ncbi/MedCPT-Query-Encoder"
    MEDCPT_ARTICLE = "MedCPT-Article-Encoder"
    
    @classmethod
    def OTHER(cls, model_path: str) -> 'EmbeddingModel':
        """Create a custom EmbeddingModel instance with an arbitrary model path"""
        # Create a new enum member dynamically
        other = object.__new__(cls)
        other._value_ = model_path
        other._name_ = f"OTHER_{model_path.replace('/', '_')}"
        return other

def load_model_from_enum(model: EmbeddingModel, float16: bool = True) -> Union[SentenceTransformer, Tuple[PreTrainedModel, PreTrainedTokenizer]]:
    """
    Factory function to load a model based on the EmbeddingModel enum
    """
    if model in [EmbeddingModel.MEDCPT_QUERY, EmbeddingModel.MEDCPT_ARTICLE]:
        return load_auto_model_tokenizer(model.value)
    model_kwargs = {"torch_dtype": "float16"} if float16 else {}
    return load_sentence_transformer_model(model.value, **model_kwargs)


class EmbeddingModelParams(BaseModel):
    
    retrival_passage: dict = Field(default_factory=dict, description="Used for passage embeddings in asymmetric retrieval tasks")
    retrival_query: dict = Field(default_factory=dict, description="Used for query embeddings in asymmetric retrieval tasks")
    separatation: dict = Field(default_factory=dict, description="Used for embeddings in clustering and re-ranking applications")
    classification: dict = Field(default_factory=dict, description="Used for embeddings in classification tasks")
    text_matching: dict = Field(default_factory=dict, description="Used for embeddings in tasks that quantify similarity between two texts, such as STS or symmetric retrieval tasks")

def load_sentence_transformer_params_from_enum(model: EmbeddingModel) -> EmbeddingModelParams:
    if model in [EmbeddingModel.JINA_EMBEDDINGS_V3]:
        """
        retrieval.query: Used for query embeddings in asymmetric retrieval tasks
        retrieval.passage: Used for passage embeddings in asymmetric retrieval tasks
        separation: Used for embeddings in clustering and re-ranking applications
        classification: Used for embeddings in classification tasks
        text-matching: Used for embeddings in tasks that quantify similarity between two texts, such as STS or symmetric retrieval tasks
        """
        return EmbeddingModelParams(
            retrival_passage={"task": "retrieval.passage"},
            retrival_query={"task": "retrieval.query"},
            separatation={"task": "separation"},
            classification={"task": "classification"},
            text_matching={"task": "text-matching"}
        )
    return EmbeddingModelParams()  # Return empty params for other models

def load_sentence_transformer_from_enum(model: EmbeddingModel, **kwargs) -> SentenceTransformer:
    """
    Factory function to load only SentenceTransformer models based on the EmbeddingModel enum.
    Raises ValueError if the model is not compatible with SentenceTransformer.
    """
    if model in [EmbeddingModel.MEDCPT_QUERY, EmbeddingModel.MEDCPT_ARTICLE]:
        raise ValueError(f"{model.name} is not compatible with SentenceTransformer")
    return load_sentence_transformer_model(model.value, **kwargs)

# You can now replace the individual loading functions with this more elegant solution
# or keep them as convenience functions that use the enum internally

if __name__ == "__main__":
    print("PRELOADING JINA EMBEDDINGS")
    model = load_sentence_transformer_from_enum(EmbeddingModel.JINA_EMBEDDINGS_V3)
    print(model.encode("Hello, world!"))
