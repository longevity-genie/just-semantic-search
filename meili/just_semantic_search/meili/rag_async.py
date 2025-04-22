from just_semantic_search.meili.rag import MeiliBase, log_retry_errors
from just_semantic_search.document import ArticleDocument, Document
from typing import List, Optional
from pydantic import Field

from meilisearch_python_sdk import AsyncClient, AsyncIndex
from meilisearch_python_sdk.errors import MeilisearchApiError
from meilisearch_python_sdk.models.settings import MeilisearchSettings, UserProvidedEmbedder

import asyncio
from eliot import start_action


    
class MeiliAsyncRAG(MeiliBase):
    # SO FOR NOT USED CAUSE IT IS UNPREDICTABLE
    #WORK IN PROGRESS
    #it will be future fixure

    index_async: Optional[AsyncIndex] = Field(default=None, exclude=True)
    

    def get_loop(self):
        """Helper to get or create an event loop that works in all environments"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    

    def run_async(self, coro):
        """Helper method to run async code safely in all environments"""
        loop = self.get_loop()
        if loop.is_running():
            # Create a new loop for this operation if the current one is running
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
                asyncio.set_event_loop(loop)
        else:
            return loop.run_until_complete(coro)


    @log_retry_errors
    async def _init_index_async(self, 
                         create_index_if_not_exists: bool = True, 
                         recreate_index: bool = False) -> AsyncIndex:
        with start_action(action_type="init_index_async") as action:
            try:
                index = await self.client_async.get_index(self.index_name)
                if recreate_index:
                    action.log(
                        message_type="index_exists",
                        index_name=self.index_name,
                        recreate_index=True
                    )
                    deleted = await self.delete_index_async()
                    index = await self.client_async.create_index(self.index_name)
                    return index
                else:
                    action.add_success_fields(
                        message_type="index_exists",
                        index_name=self.index_name,
                        recreate_index=False
                    )
                    return index
            except MeilisearchApiError:
                if create_index_if_not_exists:
                    action.add_success_fields(
                        message_type="index_not_found",
                        index_name=self.index_name,
                        create_index_if_not_exists=True
                    )
                    index = await self.client_async.create_index(self.index_name)
                    await index.update_searchable_attributes(self.searchable_attributes)
                    await index.update_filterable_attributes(self.filterable_attributes)
                    return index
                else:
                    action.log(
                        message_type="index_not_found",
                        index_name=self.index_name,
                        create_index_if_not_exists=False
                    )
            return await self.client_async.get_index(self.index_name)


    @log_retry_errors
    async def add_documents_async(self, documents: List[ArticleDocument | Document], compress: bool = False) -> int:
        """Add ArticleDocument objects to the index."""
        with start_action(action_type="add documents") as action:
            documents_dict = [doc.model_dump(by_alias=True) for doc in documents]
            count = len(documents)
            result = await self.index_async.add_documents(documents_dict, primary_key=self.primary_key, compress=compress)
            action.add_success_fields(
                status=result.status,
                count = count
            )
            return result

    def model_post_init(self, __context) -> None:
        """Initialize clients and configure index after model initialization"""
        super().model_post_init(__context)
        self.client_async = AsyncClient(base_url=f'http://{self.host}:{self.port}', api_key=self.api_key)
        self.index_async = self.client_async.index(self.index_name)

        #self.index_async = self.run_async(
        #    self._init_index_async(self.create_index_if_not_exists, self.recreate_index)
        #)
        #self.run_async(self._configure_index())

    @log_retry_errors
    async def delete_index_async(self):
        return await self.client_async.delete_index_if_exists(self.index_name)
      
    @log_retry_errors
    async def _configure_async_index(self):
        embedder = UserProvidedEmbedder(
            dimensions=1024,
            source="userProvided"
        )
        embedders = {
            self.model_name: embedder
        }
        settings = MeilisearchSettings(embedders=embedders, searchable_attributes=self.searchable_attributes)
        return await self.index_async.update_settings(settings)