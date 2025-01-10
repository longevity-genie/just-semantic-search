from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


def get_sentence_transformer_model_name(model: SentenceTransformer) -> str | None:
    for module in model.modules():
        if hasattr(module, 'auto_model'):
            return module.auto_model.name_or_path
    return None
