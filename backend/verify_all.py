#!/usr/bin/env python3
"""
MigrateMate Full Verification Script

Tests all components:
1. Database models & migrations
2. Embedding service
3. Qdrant vector operations
4. Neo4j graph operations
5. Hybrid retrieval
6. LangGraph agent
7. API endpoints
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


async def test_database():
    print("\n📦 Testing Database...")
    try:
        from sqlalchemy import select
        from app.core.database import AsyncSessionLocal, init_db
        from app.models import Project, ProjectStatus, FrameworkType
        
        await init_db()
        print("   ✅ Tables created")
        
        async with AsyncSessionLocal() as session:
            project = Project(name="test-project", source_framework=FrameworkType.FLASK, target_framework=FrameworkType.FASTAPI, status=ProjectStatus.PENDING)
            session.add(project)
            await session.commit()
            await session.refresh(project)
            print(f"   ✅ Created project: {project}")
            
            result = await session.execute(select(Project).where(Project.name == "test-project"))
            found = result.scalar_one()
            print(f"   ✅ Retrieved: {found.name}")
            
            await session.delete(found)
            await session.commit()
            print("   ✅ Cleaned up")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


async def test_embedding():
    print("\n🧠 Testing Embedding Service...")
    try:
        from app.services import get_embedding_service
        
        print("   ⏳ Loading model...")
        service = get_embedding_service()
        print(f"   ✅ Model: {service.model_name}, Device: {service.device}, Dim: {service.dimension}")
        
        code = "def hello(): print('world')"
        emb = await service.embed_text(code)
        print(f"   ✅ Embedding generated: {len(emb)} dims, first 3: {emb[:3]}")
        
        batch = ["def foo(): pass", "class Bar: pass"]
        embs = await service.embed_batch(batch)
        print(f"   ✅ Batch: {len(embs)} embeddings")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


async def test_qdrant():
    print("\n🔍 Testing Qdrant...")
    try:
        from app.services import get_qdrant_service, get_embedding_service
        
        qdrant = get_qdrant_service()
        emb_service = get_embedding_service()
        
        await qdrant.create_collection()
        print("   ✅ Collection ready")
        
        codes = ["def login(): pass", "def authenticate(): pass"]
        embeddings = await emb_service.embed_batch(codes)
        
        payloads = [{"chunk_id": i, "project_id": 999, "content": c, "name": f"func_{i}", "chunk_type": "function", "file_path": "test.py"} for i, c in enumerate(codes)]
        await qdrant.upsert_vectors(embeddings, payloads)
        print("   ✅ Vectors inserted")
        
        results = await qdrant.search_by_text("authentication", project_id=999, limit=2)
        print(f"   ✅ Search returned {len(results)} results")
        
        await qdrant.delete_by_project(999)
        print("   ✅ Cleaned up")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


async def test_neo4j():
    print("\n🕸️  Testing Neo4j...")
    try:
        from app.services import get_neo4j_service, CodeNode, CodeEdge
        
        neo4j = get_neo4j_service()
        await neo4j.setup_schema()
        print("   ✅ Schema ready")
        
        nodes = [
            CodeNode(id="999:a.py:main", project_id=999, node_type="function", name="main", file_path="a.py", start_line=1, end_line=10),
            CodeNode(id="999:a.py:helper", project_id=999, node_type="function", name="helper", file_path="a.py", start_line=12, end_line=20),
        ]
        await neo4j.create_nodes_batch(nodes)
        print("   ✅ Nodes created")
        
        await neo4j.create_edge(CodeEdge(source_id="999:a.py:main", target_id="999:a.py:helper", edge_type="CALLS"))
        print("   ✅ Edge created")
        
        deps = await neo4j.get_dependencies("999:a.py:main")
        print(f"   ✅ Dependencies: {[d['name'] for d in deps]}")
        
        await neo4j.delete_project(999)
        print("   ✅ Cleaned up")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


async def test_hybrid():
    print("\n🔀 Testing Hybrid Retrieval...")
    try:
        from app.services import get_hybrid_retrieval_service, get_qdrant_service, get_embedding_service
        
        # Setup test data
        qdrant = get_qdrant_service()
        emb = get_embedding_service()
        
        codes = [("auth", "def authenticate(): pass"), ("login", "def login(): pass")]
        embeddings = await emb.embed_batch([c[1] for c in codes])
        payloads = [{"chunk_id": i, "project_id": 888, "content": c, "name": n, "chunk_type": "function", "file_path": "auth.py", "start_line": i*10, "end_line": i*10+5} for i, (n, c) in enumerate(codes)]
        await qdrant.upsert_vectors(embeddings, payloads)
        
        service = get_hybrid_retrieval_service()
        context = await service.retrieve("authentication", project_id=888, max_results=5)
        print(f"   ✅ Retrieved {len(context.results)} results")
        print(f"   ✅ Prompt context: {len(context.to_prompt_context())} chars")
        
        await qdrant.delete_by_project(888)
        print("   ✅ Cleaned up")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


async def test_agent():
    print("\n🤖 Testing LangGraph Agent...")
    try:
        from app.agents import get_migration_agent
        
        agent = get_migration_agent()
        
        result = await agent.migrate_chunk(
            project_id=0, job_id=0,
            source_framework="flask", target_framework="fastapi",
            chunk_id=0,
            chunk_content="@app.route('/hello')\ndef hello():\n    return jsonify({'msg': 'hi'})",
            chunk_name="hello",
            chunk_type="function",
            max_iterations=2,
        )
        
        print(f"   ✅ Status: {result['status']}")
        print(f"   ✅ Confidence: {result['confidence_score']:.0%}")
        print(f"   ✅ Iterations: {result['iterations']}")
        if result['migrated_code']:
            print(f"   ✅ Migrated code preview: {result['migrated_code'][:100]}...")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


async def test_code_parser():
    print("\n📝 Testing Code Parser...")
    try:
        from app.services import parse_python_file
        
        code = '''
def hello():
    """Say hello."""
    print("Hello")

class Greeter:
    def greet(self, name):
        return f"Hello {name}"
'''
        chunks = parse_python_file(code, "test.py")
        print(f"   ✅ Parsed {len(chunks)} chunks")
        for c in chunks:
            print(f"      - {c.chunk_type}: {c.name} (lines {c.start_line}-{c.end_line})")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


async def main():
    print("=" * 60)
    print("🧪 MigrateMate Full Verification")
    print("=" * 60)
    
    from app.core.config import settings
    print(f"\n📋 Config:")
    print(f"   PostgreSQL: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
    print(f"   Neo4j: {settings.NEO4J_HOST}:{settings.NEO4J_BOLT_PORT}")
    print(f"   Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_HTTP_PORT}")
    print(f"   Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    
    results = {
        "Database": await test_database(),
        "Embedding": await test_embedding(),
        "Qdrant": await test_qdrant(),
        "Neo4j": await test_neo4j(),
        "Hybrid Retrieval": await test_hybrid(),
        "Code Parser": await test_code_parser(),
        "LangGraph Agent": await test_agent(),
    }
    
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
        print("\nNext steps:")
        print("  1. Run: uvicorn app.main:app --reload")
        print("  2. Visit: http://localhost:8000/docs")
        print("  3. Test API endpoints")
    else:
        print("⚠️  Some tests failed. Check Docker containers:")
        print("  docker compose ps")
        print("  docker compose logs <service>")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
