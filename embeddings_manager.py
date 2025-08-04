from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="document_chunks")
        self.chunks = []
    
    def create_index(self, texts: list):
        """Create vector index from list of texts"""
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Store in ChromaDB
            ids = [str(i) for i in range(len(texts))]
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                ids=ids
            )
            self.chunks = texts
            return True
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search index for similar texts"""
        try:
            # Embed the query
            query_embedding = self.model.encode([query]).tolist()
            
            # Search the collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k
            )
            
            # Format results
            formatted_results = []
            for doc, score in zip(results['documents'][0], results['distances'][0]):
                formatted_results.append({
                    "text": doc,
                    "score": float(score)
                })
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            return []