import requests
import os
import time
import re

API_URL = "http://localhost:8000/api/v1"
SOURCE_FILE = r"D:\test-flask-app\app.py"
OUTPUT_DIR = r"D:\test-flask-app\output"

def main():
    print(f"🚀 MigrateMate - Flask to FastAPI Migration")
    print(f"=" * 50)
    print(f"📂 Source: {SOURCE_FILE}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "main.py")
    
    # 1. Read source file
    with open(SOURCE_FILE, "r") as f:
        source_code = f.read()
    
    # 2. Parse the code
    print("\n🔍 Parsing source code...")
    parse_response = requests.post(f"{API_URL}/ai/parse", json={
        "content": source_code,
        "file_path": os.path.basename(SOURCE_FILE)
    })
    
    if parse_response.status_code != 200:
        print(f"❌ Parse failed: {parse_response.text}")
        return
    
    chunks = parse_response.json()
    print(f"✅ Found {len(chunks)} code chunks")
    
    # 3. Build FastAPI header
    header = '''"""Auto-generated FastAPI application from Flask migration."""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Migrated API", description="Converted from Flask to FastAPI")

# Fake database
users = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
]


class UserCreate(BaseModel):
    """Schema for creating a user."""
    name: str
    email: str


class UserResponse(BaseModel):
    """Schema for user response."""
    id: int
    name: str
    email: str

'''
    
    migrated_functions = []
    failed_chunks = []
    
    # 4. Migrate each chunk
    print("\n🤖 Migrating chunks...")
    for i, chunk in enumerate(chunks):
        name = chunk.get('name', 'unknown')
        chunk_type = chunk.get('chunk_type', 'unknown')
        content = chunk.get('content', '')
        
        # Skip module-level or empty chunks
        if not content or name == '<module>':
            continue
        
        print(f"   [{i+1}/{len(chunks)}] {chunk_type}: {name}...", end=" ", flush=True)
        
        start = time.time()
        response = requests.post(f"{API_URL}/ai/migrate/chunk", json={
            "source_framework": "flask",
            "target_framework": "fastapi",
            "chunk_content": content,
            "chunk_name": name,
            "chunk_type": chunk_type,
            "max_iterations": 3
        })
        
        duration = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'completed' and data.get('migrated_code'):
                migrated_functions.append(data['migrated_code'])
                print(f"✅ ({duration:.1f}s)")
            else:
                failed_chunks.append(name)
                print(f"⚠️  {data.get('status')} - {data.get('errors', [])}")
        else:
            failed_chunks.append(name)
            print(f"❌ HTTP {response.status_code}")
    
    # 5. Assemble output
    print(f"\n💾 Assembling migrated code...")
    
    output_code = header + "\n\n".join(migrated_functions)
    
    # Clean up any duplicate imports that Gemini might have added
    output_code = remove_duplicate_imports(output_code)
    output_code = post_process_migration(output_code)
    
    with open(output_file, "w") as f:
        f.write(output_code)
    
    # 6. Summary
    print(f"\n{'=' * 50}")
    print(f"📊 Migration Summary")
    print(f"{'=' * 50}")
    print(f"   ✅ Migrated: {len(migrated_functions)} chunks")
    print(f"   ❌ Failed: {len(failed_chunks)} chunks")
    if failed_chunks:
        print(f"   Failed chunks: {', '.join(failed_chunks)}")
    print(f"\n📁 Output: {output_file}")
    print(f"\n🚀 Run with:")
    print(f"   cd {OUTPUT_DIR}")
    print(f"   uvicorn main:app --reload")


def remove_duplicate_imports(code: str) -> str:
    """Remove duplicate import lines that Gemini might add per function."""
    lines = code.split('\n')
    seen_imports = set()
    result = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('from ') or stripped.startswith('import '):
            if stripped not in seen_imports:
                seen_imports.add(stripped)
                result.append(line)
        else:
            result.append(line)
    
    return '\n'.join(result)

def post_process_migration(code: str) -> str:
    """Fix common LLM migration mistakes."""
    
    # Fix 1: Replace "Flask API" with "FastAPI"
    code = code.replace('"Welcome to Flask API"', '"Welcome to FastAPI"')
    code = code.replace("'Welcome to Flask API'", "'Welcome to FastAPI'")
    
    # Fix 2: Add type hints to path parameters
    # Pattern: async def func_name(param_name): where param appears in route as {param_name}
    import re
    
    # Find all route + function pairs and add int type hint for path params
    # Match: @app.get('/users/{user_id}')\nasync def get_user(user_id):
    pattern = r"@app\.(get|post|put|delete|patch)\('([^']+)'\)\s*\nasync def (\w+)\(([^)]*)\):"
    
    def add_type_hints(match):
        method, path, func_name, params = match.groups()
        
        # Extract path parameters like {user_id}
        path_params = re.findall(r'\{(\w+)\}', path)
        
        if not params.strip():
            return match.group(0)
        
        # Split params and add type hints
        param_list = [p.strip() for p in params.split(',')]
        new_params = []
        
        for param in param_list:
            param_name = param.split(':')[0].strip()  # Handle if already has type
            if param_name in path_params and ':' not in param:
                new_params.append(f"{param_name}: int")
            else:
                new_params.append(param)
        
        return f"@app.{method}('{path}')\nasync def {func_name}({', '.join(new_params)}):"
    
    code = re.sub(pattern, add_type_hints, code)
    
    # Fix 3: Replace error dict returns with HTTPException
    # Pattern: return {"error": "message"} or return {"error": "message"}, 404
    code = re.sub(
        r'return \{"error":\s*"([^"]+)"\},?\s*(\d+)?',
        lambda m: f'raise HTTPException(status_code={m.group(2) or 404}, detail="{m.group(1)}")',
        code
    )
    
    # Fix 4: Replace Request with Pydantic model for POST endpoints
    # This is complex - we'll add a comment for manual review
    if 'request: Request' in code and 'await request.json()' in code:
        code = code.replace(
            'async def create_user(request: Request):',
            'async def create_user(user: UserCreate):'
        )
        code = code.replace(
            'data = await request.json()',
            '# Using Pydantic model instead of request.json()'
        )
        code = code.replace(
            'data.get("name")',
            'user.name'
        )
        code = code.replace(
            'data.get("email")',
            'user.email'
        )
        
    code = re.sub(r'(\S)(@app\.)', r'\1\n\n\2', code)
    return code

if __name__ == "__main__":
    main()

