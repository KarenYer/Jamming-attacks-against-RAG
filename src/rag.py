from dataclasses import dataclass
import numpy as np
from utils import giga_send, get_giga_content

@dataclass
class Database:
    texts: list
    embeddings: np.ndarray

@dataclass
class SimpleRAG:
    database: Database
    embedding_model: str
    prompt: str = """"""
    top_k: int = 1 
    giga_token: str = ''
    
    def cosine_similarity(self, A, B):
        return np.sum(A*B, axis=1)/(np.linalg.norm(A, axis=1)*(np.linalg.norm(B, axis=1)))
    
    def answer_query(self, query: str, top_k = None, llm: str = 'GigaChat', model_kwargs = {}, verbose = False):
        contexts, prompt = self.return_context_and_prompt(query, top_k)
        return self.query_llm(llm, prompt)

    async def answer_queries_a(self, queries, top_k = None, llm: str = 'GigaChat', model_kwargs = {}, verbose = False, return_contexts = False):
        raise NotImplementedError
    def return_context_and_prompt(self, query, top_k = None): 
        contexts = self.retrieve_context(query, top_k, self.embedding_model)
        prompt = f'Context information is below.\n{' '.join(contexts)}\n {self.prompt}\n Query:{query}\n Answer:'
        return contexts, prompt
    def query_llm(self, model, prompt, model_kwargs = {}):
        response =  giga_send(
            phrase= prompt,
            token=self.giga_token,
            profanity_check=True,
            model= model,
            temperature=0.01,
            top_p=0.6,
            n=1,
            system_prompt='',
            max_tokens=None
            )
        return get_giga_content(response)
    async def query_llm_a(self, model, prompt, model_kwargs):
        raise NotImplementedError
    def retrieve_context(self, query, top_k, embedding_model):
        embd_query = embedding_model(query, self.giga_token)
        full_embd_query = np.full((self.database.embeddings.shape[0], embd_query.shape[1]), embd_query )
        indices = np.argsort(self.cosine_similarity(full_embd_query, self.database.embeddings))[-1: -top_k - 1: -1]
        retrived_context = []
        for i in range(top_k):
            retrived_context.append(self.database.texts[indices[i]])
        return  retrived_context
            
