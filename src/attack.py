import numpy as np
from dataclasses import dataclass
import rag
from utils import giga_make_embeddings

@dataclass
class NaiveAttack:
    jamming: str
    
    def generate_retrieval(self, target) -> str:
        return target 
    def generate_jamming(self) -> str:
        return self.jamming
    def generate_malicious_document(self, database: rag.Database, target: str, giga_tok: str) -> rag.Database:
        mal_doc = f"{self.generate_retrieval(target)}" +  f" {self.generate_jamming()}"
        database.texts.append(mal_doc)
        embd = giga_make_embeddings(mal_doc, giga_tok)
        database.embeddings = np.append(database.embeddings, embd, axis=0)
        return database