#!/bin/bash

#init script for docker container, not yet used right now
# the idea is to have it to index everything in the folder



# Run the index_markdown.py script using poetry
poetry run python -m just_semantic_search_agent.index_markdown /app/data/
