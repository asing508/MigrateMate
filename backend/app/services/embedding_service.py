"""Local embedding service using sentence-transformers."""

import asyncio
from typing import List, Optional
import logging
import torch
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Local embedding service - no external API calls."""
    
    def __init__(self, model_name: str = None, device: str = None, batch_size: int = 32):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading embedding model '{self.model_name}' on {self.device}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimension: {self._dimension}")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_text_sync(self, text: str) -> List[float]:
        embedding = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=len(texts) > 100)
        return embeddings.tolist()
    
    async def embed_text(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text_sync, text)
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch_sync, texts)
    
    def preprocess_code(self, code: str, language: str = "python") -> str:
        prefix = f"[{language.upper()}] "
        lines = code.strip().split('\n')
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank:
                if not prev_blank:
                    cleaned_lines.append(line)
                prev_blank = True
            else:
                cleaned_lines.append(line)
                prev_blank = False
        return prefix + '\n'.join(cleaned_lines)


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


async def initialize_embedding_service() -> EmbeddingService:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_embedding_service)
