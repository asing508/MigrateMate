"""Hybrid retrieval combining vector search and graph traversal."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
from app.services.embedding_service import get_embedding_service
from app.services.qdrant_service import get_qdrant_service
from app.services.neo4j_service import get_neo4j_service

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single result from hybrid retrieval."""
    chunk_id: int
    content: str
    file_path: str
    name: Optional[str]
    chunk_type: str
    vector_score: float = 0.0
    graph_score: float = 0.0
    combined_score: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    source: str = "vector"


@dataclass
class RetrievalContext:
    """Full context returned for migration."""
    query: str
    results: List[RetrievalResult]
    total_tokens: int = 0
    
    def to_prompt_context(self) -> str:
        """Format results for LLM prompt."""
        sections = []
        for i, r in enumerate(self.results, 1):
            lang = r.file_path.split('.')[-1] if '.' in r.file_path else 'python'
            section = f"""
### Result {i}: {r.name or 'Unnamed'} ({r.chunk_type})
**File:** {r.file_path}:{r.start_line}-{r.end_line}
**Relevance:** {r.combined_score:.2%}

```{lang}
{r.content}
```"""
            if r.dependencies:
                section += f"\n**Dependencies:** {', '.join(r.dependencies[:5])}"
            sections.append(section)
        return "\n---\n".join(sections)


class HybridRetrievalService:
    """Combines vector and graph retrieval for comprehensive context."""
    
    def __init__(self, vector_weight: float = 0.7, graph_weight: float = 0.3, expand_depth: int = 1):
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.expand_depth = expand_depth
        self._embedding = None
        self._qdrant = None
        self._neo4j = None
    
    @property
    def embedding_service(self):
        if self._embedding is None:
            self._embedding = get_embedding_service()
        return self._embedding
    
    @property
    def qdrant_service(self):
        if self._qdrant is None:
            self._qdrant = get_qdrant_service()
        return self._qdrant
    
    @property
    def neo4j_service(self):
        if self._neo4j is None:
            self._neo4j = get_neo4j_service()
        return self._neo4j
    
    async def retrieve(self, query: str, project_id: int, max_results: int = 10, chunk_types: List[str] = None, include_deps: bool = True) -> RetrievalContext:
        """Perform hybrid retrieval."""
        logger.info(f"Hybrid retrieval for project {project_id}: '{query[:50]}...'")
        
        # Vector search
        vector_results = await self._vector_search(query, project_id, max_results * 2, chunk_types)
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        # Graph expansion
        if include_deps:
            vector_results = await self._expand_with_graph(vector_results)
        
        # Re-rank and deduplicate
        ranked = self._rerank(vector_results)
        final = self._deduplicate(ranked)[:max_results]
        
        return RetrievalContext(query=query, results=final, total_tokens=sum(len(r.content.split()) for r in final))
    
    async def _vector_search(self, query: str, project_id: int, limit: int, chunk_types: List[str] = None) -> List[RetrievalResult]:
        results = await self.qdrant_service.search_by_text(query_text=query, project_id=project_id, chunk_types=chunk_types, limit=limit)
        return [
            RetrievalResult(
                chunk_id=r["payload"].get("chunk_id", 0),
                content=r["payload"].get("content", ""),
                file_path=r["payload"].get("file_path", ""),
                name=r["payload"].get("name"),
                chunk_type=r["payload"].get("chunk_type", "other"),
                vector_score=r["score"],
                start_line=r["payload"].get("start_line", 0),
                end_line=r["payload"].get("end_line", 0),
                source="vector",
            )
            for r in results
        ]
    
    async def _expand_with_graph(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        for r in results:
            node_id = f"{r.chunk_id}:{r.file_path}:{r.name or 'unknown'}"
            try:
                deps = await self.neo4j_service.get_dependencies(node_id, self.expand_depth)
                r.dependencies = [d["name"] for d in deps[:10]]
                dependents = await self.neo4j_service.get_dependents(node_id, self.expand_depth)
                r.dependents = [d["name"] for d in dependents[:10]]
            except Exception:
                pass
        return results
    
    def _rerank(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        for r in results:
            conn_count = len(r.dependencies) + len(r.dependents)
            r.graph_score = min(conn_count / 10, 1.0)
            r.combined_score = self.vector_weight * r.vector_score + self.graph_weight * r.graph_score
        return sorted(results, key=lambda x: x.combined_score, reverse=True)
    
    def _deduplicate(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        seen = set()
        unique = []
        for r in results:
            key = (r.file_path, r.start_line, r.end_line)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique


_hybrid_service: Optional[HybridRetrievalService] = None

def get_hybrid_retrieval_service() -> HybridRetrievalService:
    global _hybrid_service
    if _hybrid_service is None:
        _hybrid_service = HybridRetrievalService()
    return _hybrid_service
