"""Neo4j graph service for code dependency mapping."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from neo4j import AsyncGraphDatabase, AsyncDriver
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CodeNode:
    id: str
    project_id: int
    node_type: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    chunk_id: Optional[int] = None


@dataclass
class CodeEdge:
    source_id: str
    target_id: str
    edge_type: str
    properties: Optional[Dict[str, Any]] = None


class Neo4jService:
    """Service for managing the code dependency graph."""
    
    def __init__(self):
        self._driver: Optional[AsyncDriver] = None
    
    async def get_driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        return self._driver
    
    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
    
    async def setup_schema(self) -> None:
        driver = await self.get_driver()
        async with driver.session() as session:
            queries = [
                "CREATE CONSTRAINT code_node_id IF NOT EXISTS FOR (n:CodeNode) REQUIRE n.id IS UNIQUE",
                "CREATE INDEX code_node_project IF NOT EXISTS FOR (n:CodeNode) ON (n.project_id)",
                "CREATE INDEX code_node_type IF NOT EXISTS FOR (n:CodeNode) ON (n.node_type)",
                "CREATE INDEX code_node_name IF NOT EXISTS FOR (n:CodeNode) ON (n.name)",
            ]
            for q in queries:
                try:
                    await session.run(q)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Schema warning: {e}")
        logger.info("Neo4j schema setup complete")
    
    async def create_node(self, node: CodeNode) -> str:
        driver = await self.get_driver()
        query = """
        MERGE (n:CodeNode {id: $id})
        SET n.project_id = $project_id, n.node_type = $node_type, n.name = $name,
            n.file_path = $file_path, n.start_line = $start_line, n.end_line = $end_line,
            n.chunk_id = $chunk_id, n.updated_at = datetime()
        RETURN n.id AS id
        """
        async with driver.session() as session:
            result = await session.run(query, id=node.id, project_id=node.project_id, node_type=node.node_type, name=node.name, file_path=node.file_path, start_line=node.start_line, end_line=node.end_line, chunk_id=node.chunk_id)
            record = await result.single()
            return record["id"]
    
    async def create_nodes_batch(self, nodes: List[CodeNode]) -> int:
        if not nodes:
            return 0
        driver = await self.get_driver()
        query = """
        UNWIND $nodes AS node
        MERGE (n:CodeNode {id: node.id})
        SET n.project_id = node.project_id, n.node_type = node.node_type, n.name = node.name,
            n.file_path = node.file_path, n.start_line = node.start_line, n.end_line = node.end_line,
            n.chunk_id = node.chunk_id, n.updated_at = datetime()
        RETURN count(n) AS count
        """
        node_data = [{"id": n.id, "project_id": n.project_id, "node_type": n.node_type, "name": n.name, "file_path": n.file_path, "start_line": n.start_line, "end_line": n.end_line, "chunk_id": n.chunk_id} for n in nodes]
        async with driver.session() as session:
            result = await session.run(query, nodes=node_data)
            record = await result.single()
            return record["count"]
    
    async def create_edge(self, edge: CodeEdge) -> bool:
        driver = await self.get_driver()
        rel_type = edge.edge_type.upper()
        query = f"""
        MATCH (source:CodeNode {{id: $source_id}})
        MATCH (target:CodeNode {{id: $target_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r.updated_at = datetime()
        RETURN source.id AS source
        """
        async with driver.session() as session:
            result = await session.run(query, source_id=edge.source_id, target_id=edge.target_id)
            record = await result.single()
            return record is not None
    
    async def create_edges_batch(self, edges: List[CodeEdge], edge_type: str = "DEPENDS_ON") -> int:
        if not edges:
            return 0
        driver = await self.get_driver()
        query = f"""
        UNWIND $edges AS edge
        MATCH (source:CodeNode {{id: edge.source_id}})
        MATCH (target:CodeNode {{id: edge.target_id}})
        MERGE (source)-[r:{edge_type.upper()}]->(target)
        SET r.updated_at = datetime()
        RETURN count(r) AS count
        """
        edge_data = [{"source_id": e.source_id, "target_id": e.target_id} for e in edges]
        async with driver.session() as session:
            result = await session.run(query, edges=edge_data)
            record = await result.single()
            return record["count"]
    
    async def get_dependencies(self, node_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        driver = await self.get_driver()
        # FIX: Use f-string for depth, NOT $depth parameter
        query = f"""
        MATCH path = (source:CodeNode {{id: $node_id}})-[*1..{depth}]->(dep:CodeNode)
        RETURN dep.id AS id, dep.name AS name, dep.node_type AS node_type, dep.file_path AS file_path, length(path) AS distance
        ORDER BY distance, dep.name
        """
        async with driver.session() as session:
            # Remove depth from the run parameters
            result = await session.run(query, node_id=node_id) 
            return await result.data()
    
    async def get_dependents(self, node_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        driver = await self.get_driver()
        # FIX: Use f-string for depth, NOT $depth parameter
        query = f"""
        MATCH path = (dependent:CodeNode)-[*1..{depth}]->(target:CodeNode {{id: $node_id}})
        RETURN dependent.id AS id, dependent.name AS name, dependent.node_type AS node_type, dependent.file_path AS file_path, length(path) AS distance
        ORDER BY distance, dependent.name
        """
        async with driver.session() as session:
            # Remove depth from the run parameters
            result = await session.run(query, node_id=node_id)
            return await result.data()
    
    async def get_project_graph(self, project_id: int, limit: int = 1000) -> Dict[str, Any]:
        driver = await self.get_driver()
        async with driver.session() as session:
            nodes_result = await session.run("MATCH (n:CodeNode {project_id: $project_id}) RETURN n.id AS id, n.name AS name, n.node_type AS type LIMIT $limit", project_id=project_id, limit=limit)
            nodes = await nodes_result.data()
            edges_result = await session.run("MATCH (s:CodeNode {project_id: $project_id})-[r]->(t:CodeNode) RETURN s.id AS source, t.id AS target, type(r) AS type LIMIT $limit", project_id=project_id, limit=limit)
            edges = await edges_result.data()
            return {"nodes": nodes, "edges": edges}
    
    async def get_migration_order(self, project_id: int) -> List[Dict[str, Any]]:
        driver = await self.get_driver()
        query = """
        MATCH (n:CodeNode {project_id: $project_id})
        OPTIONAL MATCH (n)-[:CALLS|IMPORTS|USES]->(dep:CodeNode)
        WITH n, count(dep) AS dep_count
        ORDER BY dep_count ASC, n.file_path, n.start_line
        RETURN n.id AS id, n.name AS name, n.node_type AS node_type, n.chunk_id AS chunk_id, dep_count
        """
        async with driver.session() as session:
            result = await session.run(query, project_id=project_id)
            return await result.data()
    
    async def delete_project(self, project_id: int) -> int:
        driver = await self.get_driver()
        query = "MATCH (n:CodeNode {project_id: $project_id}) DETACH DELETE n RETURN count(n) AS count"
        async with driver.session() as session:
            result = await session.run(query, project_id=project_id)
            record = await result.single()
            return record["count"]
    
    async def get_stats(self) -> Dict[str, Any]:
        driver = await self.get_driver()
        async with driver.session() as session:
            try:
                result = await session.run("MATCH (n:CodeNode) RETURN count(n) AS total_nodes")
                record = await result.single()
                nodes = record["total_nodes"] if record else 0
                result2 = await session.run("MATCH ()-[r]->() RETURN count(r) AS total_edges")
                record2 = await result2.single()
                edges = record2["total_edges"] if record2 else 0
                return {"total_nodes": nodes, "total_edges": edges}
            except Exception:
                return {"total_nodes": 0, "total_edges": 0}


_neo4j_service: Optional[Neo4jService] = None

def get_neo4j_service() -> Neo4jService:
    global _neo4j_service
    if _neo4j_service is None:
        _neo4j_service = Neo4jService()
    return _neo4j_service
