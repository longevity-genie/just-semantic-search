from just_semantic_search.splitters.abstract_splitters import AbstractSplitter, SentenceTransformerMixin
from typing import List, TypeAlias, Generic, Optional
import numpy as np
from pathlib import Path
import re
from just_semantic_search.document import ArticleDocument, Document, IDocument
from pydantic import Field

from just_semantic_search.document import Document, IDocument
from typing import Generic, List, Optional, TypeAlias
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path

class AbstractDictionarySplitter(AbstractSplitter[dict, IDocument]):
    """Implementation of AbstractSplitter for text content that works with any Document type."""

    content_key: str = "content"

class DictionarySplitter(AbstractDictionarySplitter, SentenceTransformerMixin):
    """Implementation of AbstractSplitter for text content that works with any Document type."""

    content_key: str = "content"
    
    def split(self, text: dict) -> List[dict]:
        return [text]
    