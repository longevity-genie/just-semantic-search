from typing import Union


from eliot import FileDestination
from eliottree import render_tasks, tasks_from_iterable
import json

from eliot import FileDestination
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer

def see_tokens(inputs: Union[list[str], str], model):
    if isinstance(inputs, str):
        inputs = [inputs]
    
    tokenized_data = model.tokenize(inputs)
    return model.tokenizer.convert_ids_to_tokens(tokenized_data["input_ids"][0])

def see_auto_tokens(inputs: Union[list[str], str], model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    if isinstance(inputs, str):
        inputs = [inputs]
    
    tokenized_data = tokenizer(inputs)
    return tokenizer.convert_ids_to_tokens(tokenized_data["input_ids"][0])

