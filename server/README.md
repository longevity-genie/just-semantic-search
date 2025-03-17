RAG Server package 

Depends on the just-agents package

This package also has some features which are specific for [Just-Chat](https://github.com/longevity-genie/just-chat) integration.

It allows easier index for markdown files.

1. Indexing papers

For indexing what you have to do is run
```bash
poetry shell
index_markdown <path_to_markdown_folder> -i <index_name>
```
this will index all markdown files in the folder and create a Meilisearch index. It will all be indexed in the location where you gave the input files from.
 NOTE: this is run here, in the just-semantic-search folder but can have targets outside of this folder, for example in other projects.


2. Searching indexed papers

First off- searching is mostly done in the project you are working on. 
Meaning that the primary usecase is for the user to import the libary.

2.1. Just-Agents have you should configure a web_agent in your `agent_profiles.yaml`.
You can either user meilisearch separately from just-chat or you can extend the just-chat docker-compose.yml file with the following meilisearch service.
```
meilisearch:
    container_name: meilisearch
    image: getmeili/meilisearch:v1.13.0
    environment:
      - http_proxy
      - https_proxy
      - MEILI_MASTER_KEY=fancy_master_key
      - MEILI_NO_ANALYTICS=${MEILI_NO_ANALYTICS:-true}
      - MEILI_ENV=${MEILI_ENV:-development}
      - MEILI_LOG_LEVEL
      - MEILI_DB_PATH=${MEILI_DB_PATH:-/data.ms}
      - MEILI_EXPERIMENTAL_ENABLE_METRICS=true
      - MEILI_EXPERIMENTAL_ENABLE_VECTORS=true
    ports:
      - ${MEILI_PORT:-7700}:7700
    volumes:
      - ./data.ms:/data.ms
    restart: unless-stopped
```
2.2. in `requirements.txt` you have to add `just-semantic-search-meili`
2.3. in `agent_profiles.yaml` you have to add the following tools for the agent you want to use it
```
      - package: "just_semantic_search.meili.tools"
        function: "search_documents"
```

This will allow the agent to use the search_documents function from the just_semantic_search.meili.tools package.
Also, given you have indexed the papers in this project, the only part from the libary you will use is the search_documents function.

2.4. run 
```
 docker compose up 
 ``` 
 and querry the agent so that it will have to search your indexed papers


NOTE: to check things before you run the agent you can first check port ```0.0.0.0:7700``` to see if the meilisearch is running. -key is ``fancy_master_key``
There you should be able to see whether meilisearch is running and if there are indexes created.

Following text explains more in details how this library works and it is structured

-----------------------------------------------------------
