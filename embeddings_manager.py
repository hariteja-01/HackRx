from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def create_index(self, texts: list):
        """Create FAISS index from list of texts"""
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Create and populate index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            self.chunks = texts
            return True
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 3):
        """Search index for similar texts"""
        if not self.index:
            return []
        
        try:
            # Embed the query
            query_embedding = self.model.encode([query])
            
            # Search the index
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Return results with scores
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= 0:  # -1 indicates no result
                    results.append({
                        "text": self.chunks[idx],
                        "score": float(distance)
                    })
            return results
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            return []
    
    def save_index(self, file_path: str):
        """Save FAISS index to file"""
        if not self.index:
            return False
        
        try:
            faiss.write_index(self.index, file_path)
            return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, file_path: str, chunks: list):
        """Load FAISS index from file"""
        try:
            self.index = faiss.read_index(file_path)
            self.chunks = chunks
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False